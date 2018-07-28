from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from pyspark import SparkConf

# Input data: Each row is a bag of words with a ID.

conf = SparkConf().setMaster('222.201.145.132:7077')

spark = SparkSession.builder.appName("count vec").getOrCreate()
df = spark.createDataFrame([
    (0, "a b c".split(" ")),
    (1, "a b b c a".split(" "))
], ["id", "words"])

# fit a CountVectorizerModel from the corpus.
cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)

model = cv.fit(df)

result = model.transform(df)
result.show(truncate=False)
