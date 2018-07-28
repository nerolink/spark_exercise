from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("e").getOrCreate()

training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0, Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.2, -0.5]))], ["label", "features"])

lr = LogisticRegression(maxIter=10, regParam=0.01)
model_1 = lr.fit(training)

param_map = dict()
param_map[lr.maxIter] = 30
param_map.update({lr.regParam: 0.1, lr.threshold: 0.55})

param_map_new = {lr.probabilityCol: "my_probability"}
param_map_combined = param_map.copy()
param_map_combined.update(param_map_new)

model_2 = lr.fit(training, params=param_map_combined)

test = spark.createDataFrame([
    (1.0, Vectors.dense([-1.0, 1.5, 1.3])),
    (0.0, Vectors.dense([3.0, 2.0, -0.1])),
    (1.0, Vectors.dense([0.0, 2.2, -1.5]))], ["label", "features"])

predict = model_2.transform(test)
result = predict.select("features", "label", "my_probability", "prediction")
result.show(truncate=False)
