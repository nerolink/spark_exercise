from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import ChiSquareTest
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("cs").getOrCreate()

data = [(0.0, Vectors.dense(0.5, 10.0)),
        (0.0, Vectors.dense(1.5, 20.0)),
        (1.0, Vectors.dense(1.5, 30.0)),
        (0.0, Vectors.dense(3.5, 30.0)),
        (0.0, Vectors.dense(3.5, 40.0)),
        (1.0, Vectors.dense(3.5, 40.0))]

df = spark.createDataFrame(data, schema=['label', 'features'])
r = ChiSquareTest.test(df, "features", "label").head()
print("pValues" + str(r.pValues))
print("degreeOfFreedom:" + str(r.degreesOfFreedom))
print("statistics:" + str(r.statistics))
