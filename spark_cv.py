from __future__ import print_function

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.classification import LogisticRegression, OneVsRest, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler





if __name__ == "__main__":
    conf = SparkConf(True)
    conf.set("spark.executor.memory", "32g")
    conf.set('spark.executor.cores', '4')

    sc = SparkContext(
        appName="multilayer_perceptron_classification_example",
        conf=conf
    )

    sqlContext = SQLContext(sc)


    train = sqlContext.read.format('com.databricks.spark.csv').options(header = True, inferschema = True, sep = ",").load("mnist_data/train.csv")
    features = train.schema.names[1:]
    vectorizer = VectorAssembler(inputCols=features, outputCol="features")

    training = vectorizer.transform(train).select("label", "features")
    print(training.printSchema())


    print(training.show(5))

    train, test = training.randomSplit([0.8,0.2], 1)

    
    lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True)
    ovr = OneVsRest(classifier=lr)

    layers = [28*28, 1024, 10]
    mlp = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=0)

    print("---Training---")

    print("LR:")
    model_lr = ovr.fit(train)
    model_mlp = mlp.fit(train)
    print("MLP:")
    print("---Testing---")

    result_lr = model_lr.transform(test)
    result_mlp = model_lr.transform(test)

    predictionAndLabels_lr = result_lr.select("prediction", "label")
    predictionAndLabels_mlp = result_mlp.select("prediction", "label")

    evaluator = MulticlassClassificationEvaluator()

    print("Precision LR: " + str(evaluator.evaluate(predictionAndLabels_lr)))
    print("Precision MLP: " + str(evaluator.evaluate(predictionAndLabels_mlp)))

    