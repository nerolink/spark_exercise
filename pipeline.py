from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("pipe line").getOrCreate()

training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

token = Tokenizer(inputCol='text', outputCol='words')
hashing_TF = HashingTF(inputCol=token.getOutputCol(), outputCol='features')
lr = LogisticRegression(maxIter=10, regParam=0.0001)
pipeline = Pipeline(stages=[token, hashing_TF, lr])

model = pipeline.fit(training)

test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

prediction = model.transform(test)
selection = prediction.select("id", "text", "probability", "prediction")
for row in selection.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))
