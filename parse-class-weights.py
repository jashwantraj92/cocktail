import csv
import operator
from collections import OrderedDict
from collections import defaultdict
weights = defaultdict(lambda: defaultdict(int))
with open('1596592079.0497162-150.0-0.82-250.0-output.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        model = rows[0]
        Class = rows[1]
        predictions = rows[2]
        weights[model][Class]=predictions   
w = csv.writer(open('class-weights.csv', "w"))
for key,val in weights.items():
    #print(val)
    val = OrderedDict(sorted(val.items(), key=lambda x:x[1], reverse=True))
    weights[key] = val
    total = 0
    for k,v in val.items():
        total+=int(v)
        w.writerow([key, k, v])
    print(key," accuracy is ",total,total/50000*100)
    #print(key,val)
    #weights[key] = sorted(weights[key].items(), key=lambda x: weights[key][x]

