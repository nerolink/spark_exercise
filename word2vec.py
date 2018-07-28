from pyspark.ml.feature import Word2Vec
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("word to vec").getOrCreate()

# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("Hi I heard about Spark".split(" "),),
    ("I wish Java could use case classes".split(" "),),
    ("Logistic regression models are neat".split(" "),)
], ["text"])

word2vec = Word2Vec(vectorSize=3, minCount=0, inputCol='text', outputCol='result')
model = word2vec.fit(documentDF)
result = model.transform(documentDF)
for row in result.collect():
    text, vector = row
    print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))
