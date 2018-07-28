from pyspark.sql import SparkSession
from pyspark.ml.feature import FeatureHasher

spark = SparkSession.builder.appName("hash").getOrCreate()
data_set = spark.createDataFrame([
    (2.2, True, "1", "foo"),
    (3.3, False, "2", "bar"),
    (4.4, False, "3", "baz"),
    (5.5, False, "4", "foo")
], ["real", "bool", "stringNum", "string"])

hash_er = FeatureHasher(inputCols=["real", "bool", "stringNum", "string"], outputCol="feature")
features = hash_er.transform(data_set)
print(type(features))
features.show(truncate=False)
