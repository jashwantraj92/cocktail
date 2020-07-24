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
if not sys.argv[1]:
	print("please enter view or remove")
	sys.exit()
option = sys.argv[1]
collections=["instance","aws","pre_aws","on_demand","spot"]
if option == "view":
	print(list(mydb["instance"].find()))
	print(len(list(mydb["instance"].find())))
	print(list(mydb["instance"].find()))
	print(len(list(mydb["aws"].find())))

elif option =="remove":
	for name in collections:
		mycol = mydb[name]
		mycol.delete_many({})
elif option == "ips":
	records = list(mydb["instance"].find())
	ips = []
	IPS = ""
	for i in range(len(records[0]['instances'])): 
		print(records[0]['instances'][i]['ip'])
		IPS= IPS + " " + (records[0]['instances'][i]['ip'].strip("'"))
	#allips = [IPS + " " + i.strip("'") for i in ips]
	print(IPS)
        
else:
	print("please enter view or remove")

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
