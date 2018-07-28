from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark import SparkConf

spark = SparkSession.builder.appName(
    "CorrelationExample").getOrCreate()

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),), (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        # (c1,c2,c3,..)圆括号里的是column
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),), (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
# df = spark.createDataFrame(data, ["features"])  # 每一行都是features
df = spark.createDataFrame(data, ['features'])
print(df.collect())

r1 = Correlation.corr(df, "features").head()
print("Pearson correlation matrix:\n" + str(r1[0]))

r2 = Correlation.corr(df, "features", "spearman").head()
print("Spearman correlation matrix:\n" + str(r2[0]))
# $example off$

spark.stop()
