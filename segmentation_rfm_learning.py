from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from spark.python.pyspark.sql.functions import col

# Load and preprocess the data
df = rfmMap.withColumn('segment', rfmMap['RFM_SCORE'])  # create new column for segment

# Split the data
(trainingData, testData) = df.randomSplit([0.7, 0.3], seed=1234)
trainingData = trainingData.withColumn("RFM_SCORE", col("RFM_SCORE").cast("double"))
# Define feature vector

assembler = VectorAssembler(inputCols=['RFM_SCORE'], outputCol='features')

# Define classifier
dt = DecisionTreeClassifier(labelCol='segment', featuresCol='features')

# Define pipeline
pipeline = Pipeline(stages=[assembler, dt])

# Train model
model = pipeline.fit(trainingData)

# Evaluate model
predictions = model.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol='segment', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print('Accuracy = {}'.format(accuracy))

# Use model to classify new data
newData = spark.read.csv('new_rfm_scores.csv', header=True, inferSchema=True)
predicted = model.transform(newData)
predicted.show()
