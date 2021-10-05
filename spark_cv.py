from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from functools import reduce

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
#from sparkdl import DeepImageFeaturizer

# create a spark session
spark = SparkSession.builder.appName('DigitRecog').getOrCreate()
# loaded image
zero = spark.read.format("image").load("mnist_data/0")
one = spark.read.format("image").load("mnist_data/1")
two = spark.read.format("image").load("mnist_data/2")
three = spark.read.format("image").load("mnist_data/3")
four = spark.read.format("image").load("mnist_data/4")
five = spark.read.format("image").load("mnist_data/5")
six = spark.read.format("image").load("mnist_data/6")
seven = spark.read.format("image").load("mnist_data/7")
eight = spark.read.format("image").load("mnist_data/8")
nine = spark.read.format("image").load("mnist_data/9")

print(zero.printSchema())
dataframes = [zero, one, two, three,four,
             five, six, seven, eight, nine]
# merge data frame
df = reduce(lambda first, second: first.union(second), dataframes)
# repartition dataframe 
df = df.repartition(200)
# split the data-frame
train, test = df.randomSplit([0.8, 0.2], 42)

print(train.show(5))

'''
featurizer = DeepImageFeaturizer(inputCol="image",
                                 outputCol="features",
                                 modelName="InceptionV3")
'''

lr = LogisticRegression(maxIter=5, regParam=0.03, 
                        elasticNetParam=0.5, labelCol="label")
# define a pipeline model
sparkdn = Pipeline(stages=[ lr])
spark_model = sparkdn.fit(train)


evaluator = MulticlassClassificationEvaluator() 
tx_test = spark_model.transform(test)
print('F1-Score ', evaluator.evaluate(tx_test, 
                                      {evaluator.metricName: 'f1'}))
print('Precision ', evaluator.evaluate(tx_test,
                                       {evaluator.metricName:                    'weightedPrecision'}))
print('Recall ', evaluator.evaluate(tx_test, 
                                    {evaluator.metricName: 'weightedRecall'}))
print('Accuracy ', evaluator.evaluate(tx_test, 
                                      {evaluator.metricName: 'accuracy'}))