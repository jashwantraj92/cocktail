import csv
import operator

from collections import defaultdict
weights = defaultdict(lambda: defaultdict(int))
with open('1596592079.0497162-150.0-0.82-250.0-output.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        model = rows[0]
        Class = rows[1]
        predictions = rows[2]
        weights[model][Class]=predictions   
for key in weights.keys():
    weights[key] =(sorted(weights[key].items(), key=operator.itemgetter(1)))
    #weights[key] = sorted(weights[key].items(), key=lambda x: weights[key][x])
print(weights['MobileNetV2'])
