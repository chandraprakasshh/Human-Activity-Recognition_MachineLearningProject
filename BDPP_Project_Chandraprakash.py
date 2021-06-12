#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pyspark.sql import SparkSession

spark = SparkSession     .builder     .appName("BDPPProject")     .config("spark.some.config.option", "some-value")     .getOrCreate()
       


# ## Data Load

# In[ ]:


project = spark.read.load("gs://dataproc-staging-europe-north1-1036817767992-g5iisz7g/BDProject.csv",format="csv", sep=",", inferSchema="true", header="true")


# In[ ]:


project.show(5)


# In[ ]:


project.printSchema()


# In[ ]:


project.count()


# ## Statastical Summary

# In[ ]:


project.describe().toPandas().transpose()


# In[ ]:


project.describe(['SNO', 'x_acceleration', 'y_acceleration', 'z_acceleration', 'label']).show()


# ## Data Cleaning

# In[ ]:


from pyspark.sql.functions import *
from pyspark.sql.functions import when, count, col

project.select([count(when(isnan(c)|col(c).isNull(),c)).alias(c) for c in project.columns]).show()


# In[ ]:


##Features Scaling using StandardScaler Estimator

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

assembler = VectorAssembler(inputCols = ['SNO','x_acceleration','y_acceleration','z_acceleration','label'], outputCol="features")


featuredproject = assembler.transform(project)

scaler = StandardScaler(withMean=True, withStd=True, inputCol="features", outputCol="scaledFeatures")

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(featuredproject)

# Normalize each feature to have unit standard deviation.
scaledproject = scalerModel.transform(featuredproject)
scaledproject.select(["features", "scaledFeatures"]).show(5)


# In[ ]:


from pyspark.ml.feature import OneHotEncoderEstimator

encoder = OneHotEncoderEstimator(inputCols=["label"],outputCols=["label_info"])
model = encoder.fit(project)
oheproject = model.transform(project)
oheproject.select("label_info").take(5)


# In[ ]:


from pyspark.ml import Pipeline

numPipeline = Pipeline(stages = [assembler, scaler])
catPipeline = Pipeline(stages = [encoder])
pipeline = Pipeline(stages=[numPipeline, catPipeline])
newproject = pipeline.fit(project).transform(project)
newproject.show(5)


# In[ ]:


va2 = VectorAssembler(inputCols=["scaledFeatures", "label_info"], outputCol='final_features')

temp1 = va2.transform(newproject)

dataset = temp1.withColumn('features', temp1.final_features).select("features","label")
dataset.show(5)


# ## Splitting Dataset to train and test Set

# In[ ]:


##Splitting the data in training and Testing dataset

trainSet, testSet = dataset.randomSplit([0.8, 0.2], seed=12345)
print("Training Data Count: " + str(trainSet.count()))
print("Test Data Count: " + str(testSet.count()))


# In[ ]:


trainSet.groupby('label').agg({'label': 'count'}).show()


# In[ ]:


testSet.groupby('label').agg({'label': 'count'}).show()


# ## Modelling Dataset

# ## Logistic Regression

# In[ ]:


import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression
from time import *
start_time = time()
lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=5)
lrModel = lr.fit(trainSet)
trainingSummary = lrModel.summary
trainaccuracy = trainingSummary.accuracy
print("Coefficients: %s" % str(lrModel.coefficientMatrix))
print("Intercept: %s" % str(lrModel.interceptVector))
print("Training accuracy: ",trainaccuracy)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# In[ ]:


LR_Prediction = lrModel.transform(testSet)
LR_Prediction.select('label', 'features').show(5)


# In[ ]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from time import *
start_time = time()
evaluator = lrModel.evaluate(testSet)
bc_evaluator = BinaryClassificationEvaluator()
test_roc = bc_evaluator.evaluate(LR_Prediction)
testaccuracy = evaluator.accuracy
print("Test accuracy for Logistic Regression model: ",testaccuracy)
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# ## Decision Tree Classifier

# In[ ]:


from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from time import *
start_time = time()
dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 5)
dtModel = dt.fit(trainSet)

predictions = dtModel.transform(testSet)
predictions.select('label', 'prediction', 'probability').show(5)

bt_evaluator = BinaryClassificationEvaluator()
bt_test = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = bt_test.evaluate(predictions)
print("Test accuracy for DT Model: ",testaccuracy )
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# ## Random Forest Classifier

# In[ ]:


from pyspark.ml.classification import RandomForestClassifier
from time import *
start_time = time()
randomforest = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
randomforestmodel = randomforest.fit(trainSet)

predictions = randomforestmodel.transform(testSet)
predictions.select('label', 'prediction', 'probability').show(5)
rf_evaluator = BinaryClassificationEvaluator()
test = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = test.evaluate(predictions)
print("Test accuracy for RF Model: ",testaccuracy )
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# ## Model Tunning

# In[ ]:


## Hyperparameter Tunning RF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from time import *
start_time = time()

Random_Forest_Classi = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees= 6)
pipeline = Pipeline(stages=[Random_Forest_Classi])

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=6)  


cvModel = crossval.fit(trainSet)

Prediction_Result = cvModel.transform(testSet)
Prediction_Result.select('label', 'prediction', 'probability').show(5)


BinaryClassi_evaluator = BinaryClassificationEvaluator()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = evaluator.evaluate(Prediction_Result)

print("Test accuracy For RF Model: ",testaccuracy )
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# In[ ]:


## Hyper Tunning Logistic Regression 

from time import *
start_time = time()
paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.1, 0.01])     .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=5)  


cvModel = crossval.fit(trainSet)

Prediction_Result = cvModel.transform(testSet)
Prediction_Result.select('label', 'prediction', 'probability').show(5)


BinaryClassi_evaluator = BinaryClassificationEvaluator()

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = evaluator.evaluate(Prediction_Result)

print("Test accuracy For LR Model after tunning: ",testaccuracy )
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# In[ ]:


## Hypertunning DT
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from time import *
start_time = time()
paramGrid = ParamGridBuilder()     .addGrid(lr.regParam, [0.1, 0.01])     .build()

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=BinaryClassificationEvaluator(),
                          numFolds=8)  

cvModel = crossval.fit(trainSet)

Prediction_Result = cvModel.transform(testSet)
Prediction_Result.select('label', 'prediction', 'probability').show(5)


BinaryClassi_evaluator = BinaryClassificationEvaluator()
test_roc = BinaryClassi_evaluator.evaluate(predictions, {bc_evaluator.metricName: "areaUnderROC"})
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
testaccuracy = evaluator.evaluate(Prediction_Result)

print("Test accuracy For DT Model after tunning: ",testaccuracy )
end_time = time()
elapsed_time = end_time - start_time
print("Time to train model: %.3f seconds" % elapsed_time)


# ## Uploading output file in GCP 

# In[ ]:


import random
random1 = random.randint(0,20)
filepath = "gs://dataproc-staging-europe-north1-1036817767992-g5iisz7g/" +str(random1)

df = preductions.toPandas()
csv = df.to_csv(filepath+".csv")
print("Download file name", filepath)

