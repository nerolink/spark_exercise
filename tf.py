from pyspark.mllib.feature import HashingTF
from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext

conf = SparkConf()
conf.set("spark.cores.max", "4").set("spark.executor.memory", "1g").set("spark.master",
                                                                        "spark://222.201.145.132:7077").set(
    "spark.dynamicAllocation.enabled", "false")
sc = SparkContext(conf=conf)

sentence = "hello world hello test"
words = sentence.split(" ")
tf = HashingTF(10000)  # 构建一个向量，S=10000
vec1 = tf.transform(words)
print(vec1)

rdd = sc.wholeTextFiles("file:///home/hadoop/code/spark/files").map(lambda content: content[1].split(" "))
vec2 = tf.transform(rdd)  # 对整个RDD对象进行转换，生成TF
print(vec2.collect())
