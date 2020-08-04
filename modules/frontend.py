# coding: utf-8
import asyncio
import json
import logging
import threading
import time,os

from sanic import Sanic
from sanic.response import json
from collections import defaultdict

from . import utils
from . import query_processor
from . import scheduler
from .instance_source import ins_source
from .constants import *
# from .query_processor import processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

app = Sanic(__name__)
processor = query_processor.QueryProcessor()
sch = scheduler.Scheduler()
matched = 0
not_matched = 0
images = defaultdict(list)
file1 = open('/home/cc/ensembling/CYAN/ground-truth-classes', 'r') 
lines = file1.readlines()
for line in lines:
    label = line.split(" ")[1].strip('\n')
    label = label.strip('\t')
    name = line.split(" ")[0]
    images[name].append(label)
#logging.info(images)
constraints = defaultdict(list)
for i in range(len(accuracy)):
    constraints[i].append([])

def check_ground_truth(imgcls,imgname):
    global matched,not_matched
    for i in range(len(imgcls)):
        if imgcls[i] == "resources":
		return
        if imgcls[i] == images[imgname][0]:
            logging.info(f'ground truth matched {matched} {imgcls} {imgname} {images[imgname]}')
            matched+=1
            logging.info(f"Prediction accuracy {matched/(matched+not_matched)*100}")       
            return
    not_matched+=1
    logging.info(f'ground truth not matched {not_matched} {imgcls} {imgname} {images[imgname]}')
    logging.info(f"Prediction accuracy {matched/(matched+not_matched)*100}")       

@app.route('/predict/<model_name>',  methods=['POST'])
async def predict(request, model_name):
    if request.method == 'POST':
        receive_time = utils.now()
        logging.info(f'Received request for model: {model_name}')

        typ = request.json['type']
        # use a general decoder to handle different types
        # data =  utils.decode_image(request.json['data']) if typ == 'image' else request.json['data']
        data = request.json['data']
        constraint = request.json['constraint']
        filename = request.json['filename'].strip('.JPEG')
        #constraint = 0
        #logging.info("data is ",data)
        print("data is ",len(data))
        sch.record_request(model_name)
        name , synset , typ, handel_time, models = await processor.send_query(model_name, receive_time, data, constraint)
        logging.info(f'Processed request for model: {model_name} {name} {synset}  {filename} {models}')
        check_ground_truth(synset,filename)
        if (typ > 3):
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999 + 0.001
        elif (handel_time > UPPER_LATENCY_BOUND):
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999 +  0.001
        else:
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999

        if sch.failed_rate > SLA_BOUND:
            sch.launch_standby('c5.xlarge', 1, model_name)
            scheduler.Scheduler.failed_rate = 0.0

        print("response received",name,synset) 
        logging.info(f'Model: {model_name}; typ: {typ}; handel_time: {handel_time} ; {models}; failed_rate: {scheduler.Scheduler.failed_rate}')
        return json({
            'res' : name,
            'latency' : handel_time
        })

@app.listener('after_server_start')
async def notify_server_started(app, loop):
    logging.info('enter after_server_start')
    processor.set_loop(loop)
    sch.set_loop(loop)
    ins_source.set_loop(loop)

def main(port_, tag_):
    scheduler.Tag = tag_
    app.run(port=port_, access_log=False)
