from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import sys
import json
#sc = SparkContext(master, appName)
def getTweets(rdd):
	
	x = rdd.collect()
	if(len(x)>0):
		df = spark.createDataFrame(json.loads(x[0]).values() , schema = ["sentiment_value" , "tweet"]) #creates the dataframe
		df.show(truncate=False)
		#print(json.loads(x[0]).values())
	
if __name__ == "__main__":
	
	spark = SparkSession.builder.master("local[2]").appName('GetTweet').getOrCreate()
	ssc = StreamingContext(spark.sparkContext,1)
	sqlContext = SQLContext(spark)
#lines = spark.readStream.format("socket").option("host","localhost").option("port", 6100).load()
#sqlContext = SQLContext(spark)
	lines = ssc.socketTextStream("localhost",6100)
	words = lines.flatMap(lambda line: line.split("\n"))
	words.foreachRDD(getTweets)
	
	ssc.start() # Start the computation
	ssc.awaitTermination()