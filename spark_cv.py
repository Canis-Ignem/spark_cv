from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.mllib.linalg import DenseVector, VectorUDT
from pyspark.sql import SQLContext

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, ArrayType





if __name__ == "__main__":
    conf = SparkConf(True)
    conf.set("spark.executor.memory", "8g")

    sc = SparkContext(
        appName="multilayer_perceptron_classification_example",
        conf=conf
    )

    sqlContext = SQLContext(sc)

    # train = data_frame_from_file(sqlContext, "mnist_train.csv", 0.01)
    # test = data_frame_from_file(sqlContext, "mnist_test.csv", 0.01)

    train = sqlContext.read.format('com.databricks.spark.csv').options(header = False, inferschema = True, sep = ",").load("mnist_data/train.csv")
    columns = train.schema.names
    print(columns)
    print(train.printSchema())

    layers = [28*28, 1024, 10]

    # create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(maxIter=10, layers=layers, blockSize=128, seed=1234)
    # train the model
    model = trainer.fit(train)
    # compute precision on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="precision")
    print("Precision: " + str(evaluator.evaluate(predictionAndLabels)))

    sc.stop()