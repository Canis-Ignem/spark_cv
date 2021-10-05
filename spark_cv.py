from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from functools import reduce
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