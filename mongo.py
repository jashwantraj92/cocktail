import pymongo
import time 
import sys
import datetime
import pandas as pd
import pprint
from bson.son import SON
myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["serving"]
print(mydb.list_collection_names())
print(list(mydb["instance"].find()))
print(len(list(mydb["instance"].find())))
print(list(mydb["instance"].find()))
print(len(list(mydb["aws"].find())))

mycol = mydb["job_stats"]
arrivals = []
for x in list(mycol.find()):
    arrivals.append(x['arrivalTime'])
#print(arrivals)
pprint.pprint(list(mycol.aggregate( [{ "$group": {"_id": "$container", "count": { "$sum": 1 } } }, 
				{"$sort": SON([("count", -1), ("_id", -1)])}
				])))
name="asr-slackprediction-01e7e831w881nsq2s9n1s5v7kb"
print((list(mycol.find({"container": name}))))
"""

predictor = Predictor(init_load=50,model_path='poisson_model_32.h5' , scaler_path='poisson_scaler.save')

df = pd.DataFrame((arrivals),columns=['time'])
print(df)
df.index = pd.to_datetime(df.time, unit='s' )
arrivalRate = df.groupby(df.index.second).count()
#print ("arrivalrate list is " , arrivalRate)
#print("****", df.groupby(df.index.second))
y = arrivalRate.time.tolist()
print(y)
#df.groupby(pd.Grouper(key='time', freq='1s'))
Times = 1.5
#train = df[-60:]
#print train
#	X = train.time.tolist()
#	y = train.req.tolist()
load = max(y)

forecasts = predictor.predict(load * Times)
print(forecasts)"""
