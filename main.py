import numpy as np
import pandas as pd
import os
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
#from pyspark.sql.types import StructField,StructType,StringType,LongType
from pyspark.sql.types import *
from pyspark.sql.functions import when

if __name__ == "__main__":
	spark = SparkSession.builder.appName("Regression").getOrCreate()
	
	
	emp_RDD = spark.sparkContext.emptyRDD()
	
	assembler = VectorAssembler(inputCols = ['Time','Distance','Count'],outputCol="Features")
	lr = LogisticRegression(featuresCol="Features",labelCol='Fault')
	year = [2020,2021]
	#month = [1,2,3,4,5,6,7,8,9,10,11,12]
	month = [1,2,3,4,5,6,7,8,9,10,11,12]
	schema = StructType([
		StructField("ID",StringType(),True),
		StructField("Time",LongType(),True),
		StructField("Distance",LongType(),True),
		StructField("Count",LongType(),True),
		StructField("Fault",LongType(),True)
	])
	
	df = spark.createDataFrame(emp_RDD,schema)


	for y in year:
		for m in month:
			user = spark.read.load(f"data{y}/data{y}_user/{y}_{m}.csv", format = "csv", sep=",",inferSchema="true",header="true")
			user = user.drop("_c0")
			fault = spark.read.load(f"data{y}/data{y}_fault/{y}_{m}.csv", format = "csv", sep=",",inferSchema="true",header="true")
			fault = fault.drop("_c0")
			fault = fault.withColumnRenamed("Count", "Fault")
			fault = fault.withColumn("Fault", when(col("Fault")>0, 1).otherwise(0))
			
			
			data = user.join(fault, on="ID", how="left")
			data = data.na.fill(value=0,subset=["Fault"])
			data = data.withColumn("Fault",data.Fault.cast(IntegerType()))
			
			tmp = df.union(data)
			tmp = tmp.na.fill(value=0,subset="Fault")
			tmp = tmp.groupBy("ID").sum("Time", "Distance", "Count", "Fault")
			tmp = tmp.withColumnRenamed("sum(Time)", "Time")
			tmp = tmp.withColumnRenamed("sum(Distance)", "Distance")
			tmp = tmp.withColumnRenamed("sum(Count)", "Count")
			tmp = tmp.withColumnRenamed("sum(Fault)", "Fault")
			tmp = tmp.withColumn("Fault", when(col("Fault")>0, 1).otherwise(0))
			df = tmp
			
			output = assembler.transform(df)
			
			train = output.select('Features','Fault')		
			lrn = lr.fit(train)
	lrn_summary = lrn.summary
	lrn_summary.predictions.show()		
