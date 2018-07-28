from pyspark import SparkContext
from pyspark.sql import HiveContext

sc = SparkContext()
file = sc.textFile("ExtractFile.py")
blank_lines = sc.accumulator(0)


def extract_blank_line(line):
    global blank_lines
    blank_lines += 1
    if '\n' in line:
        print("===========")
    return line.split(" ")


callSigns = file.flatMap(f=extract_blank_line)
callSigns.saveAsTextFile("output")
print(callSigns.collect())
print(blank_lines.value)
