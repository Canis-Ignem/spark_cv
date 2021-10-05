from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler





if __name__ == "__main__":
    conf = SparkConf(True)
    conf.set("spark.executor.memory", "8g")
    conf.set('spark.executor.cores', '2')

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

    train, test = training.randomSplit([0.8,0.2], 1)

    
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)

    # instantiate the One Vs Rest Classifier.
    ovr = OneVsRest(classifier=lr)

    print("---Training---")
    model = lr.fit(train)

    result = model.transform(test)
    print("---Testing---")
    predictionAndLabels = result.select("prediction", "label")

    evaluator = MulticlassClassificationEvaluator(metricName="precision")

    print("Precision: " + str(evaluator.evaluate(predictionAndLabels)))

    