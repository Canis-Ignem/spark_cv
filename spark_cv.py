spark.sparkContext.uiWebUrl
df_training = (spark
               .read
               .options(header = False, inferSchema = True)
               .csv("data/MNIST/mnist_train.csv"))

df_training.count()