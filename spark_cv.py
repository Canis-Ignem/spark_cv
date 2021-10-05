spark.sparkContext.uiWebUrl
df_training = (spark
               .read
               .options(header = False, inferSchema = True)
               .csv("/user/keystone/mnist_data/train.csv"))

df_training.count()