from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import DenseVector, VectorUDT
from pyspark.sql import SQLContext

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType

from pyspark.ml.feature import VectorAssembler





if __name__ == "__main__":
    conf = SparkConf(True)
    conf.set("spark.executor.memory", "8g")

    sc = SparkContext(
        appName="multilayer_perceptron_classification_example",
        conf=conf
    )

    sqlContext = SQLContext(sc)


    train = sqlContext.read.format('com.databricks.spark.csv').options(header = True, inferschema = True, sep = ",").load("mnist_data/train.csv")
    features = train.schema.names[1:]
    vectorizer = VectorAssembler(inputCols=features, outputCol="features")

    training = (vectorizer
            .transform(train)
            .select("label", "features")
            .toDF("label", "features")
            .cache())

    print(training.show(5))

    train, test = training.randomSplit([0.8,0.2], 0)

    
    layers = [28*28, 1024, 10]

    mlp = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)

    model = mlp.fit(train)

    result = model.transform(test)

    predictionAndLabels = result.select("prediction", "label")

    evaluator = MulticlassClassificationEvaluator(metricName="precision")

    print("Precision: " + str(evaluator.evaluate(predictionAndLabels)))

    