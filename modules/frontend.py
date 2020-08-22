# coding: utf-8
import asyncio
import json
import logging
import threading
import time,os

from sanic import Sanic
from collections import defaultdict
from sanic.response import json

from . import utils
from . import query_processor
#from .query_processor import get_correct_predictions
from . import scheduler,naiveSchedule
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
overall_accuracy = 0
step_accuracy = 0
last_time = time.time()
images = defaultdict(list)
file1 = open('/home/cc/cocktail/CYAN/ground-truth-classes', 'r') 
lines = file1.readlines()
for line in lines:
    label = line.split(" ")[1].strip('\n')
    label = label.strip('\t')
    name = line.split(" ")[0]
    images[name].append(label)
#logging.info(images)
constraints = defaultdict(list)
served_requests = defaultdict(list)
#for i in range(len(accuracy)):
#    constraints[i].append([])

def check_ground_truth(imgcls,imgname, constraint):
    global matched,not_matched
    for i in range(len(imgcls)):
        if imgcls[i] == "resources":
            return
        if imgcls[i] == images[imgname][0]:
            logging.info(f'ground truth matched {matched} {imgcls} {imgname} {images[imgname]} {constraints[constraint]}')
            matched+=1
            constraints[constraint][2]+=1
            constraints[constraint][3]+=1
            logging.info(f"Prediction accuracy {latency[constraint]} {accuracy[constraint]} {matched/(matched+not_matched)*100}")                   
            return
    not_matched+=1
    logging.info(f'ground truth not matched {not_matched} {imgcls} {imgname} {images[imgname]}')
    logging.info(f"Prediction accuracy {matched/(matched+not_matched)*100}")       

def find_model(rem_models,constraint):
		slo_latency =  latency[constraint]
		slo_accuracy = accuracy[constraint]
		candidate_models = []
		print("finding in models", rem_models, models)
		for model in rem_models:
				if latency[models.index(model)] <=  slo_latency + latency_margin:
					candidate_models.append([latency.index(model),accuracy[models.index(model)]])
		if not candidate_models:
			return None
		print(candidate_models)
		if candidate_models:
			model = max(candidate_models, key=lambda x:x[1])
			print(model, models[model[0]])
			return [models[model[0]]]

def vote_based_scaling(step_accuracy,overall_accuracy,correct_predictions,pretrained_model_list, constraint):
        slo_accuracy = accuracy[constraint]
        print("vote-based_scaling " ,step_accuracy, overall_accuracy, (slo_accuracy + 0.02)*100, latency[constraint], accuracy[constraint] ,constraints[constraint][1],constraints[constraint][2],constraints[constraint][3])
        for key in correct_predictions.keys():
                print(key,len(correct_predictions[key]))    
        if ((step_accuracy) >= ((slo_accuracy + 0.02)*100)) and len(correct_predictions) > 1:
                index,drop_model = min((len(correct_predictions[key]),key) for key in correct_predictions )
                drop = [e for e in pretrained_model_list if e == drop_model]
                print("least model is " ,index, drop_model,drop)
                pretrained_model_list.remove(drop[0])
                constraints[constraint][0].remove(drop[0])
                #union_model_list.remove(drop_model)
                del correct_predictions[drop_model]
                return drop 
        """elif ((step_accuracy) <= ((slo_accuracy - 0.02)*100)):
                remaining_models = set(models) - set(pretrained_model_list)
                print("remaining_models",remaining_models,set(models),set(pretrained_model_list[0]))
                if remaining_models:
                    model = find_model(remaining_models, constraint)
                    if model:
                        print("model added", model)
                        cmd = 'tf.keras.applications.' + str(model[0]) + '()';
                        pretrained_model = eval(cmd)
                        pretrained_model_list.append([model[0],pretrained_model])
                        return model
                    else:
                        print("***no model available to add *********")
                        return "None" """
async def get_models(constraints):
    print('main invoked') 
    accuracy,latency = await get_requirements(constraints)
    models = naiveSchedule.select_models(latency,accuracy,2,scheme)
    logging.info(f"selected models are {models}")
    return models

async def get_requirements(constraint):
    logging.info(f'constraint are{accuracy[constraint],latency[constraint]}')
    return float(accuracy[constraint]),latency[constraint]

@app.route('/predict/<model_name>',  methods=['POST'])
async def predict(request, model_name):
    global last_time
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
        logging.info(f"data is ,{constraint},{filename},{typ},")
        print("data is ",len(data))
        sch.record_request(model_name)
        if not constraints[constraint]:
            models = await get_models(int(constraint))
            constraints[constraint]=[models,1,0,0]
            logging.info(f"added models for first time {constraints[constraint]} {constraints[constraint][0]}")
            #logging.info(f"adding models for first time")
            name , synset , typ, handel_time, models = await processor.send_query(model_name, receive_time, data, constraint,filename,models)
        else:
            models = constraints[constraint][0]
            logging.info(f"updating models {constraints[constraint]}")
            constraints[constraint][1]+=1
            name , synset , typ, handel_time, models = await processor.send_query(model_name, receive_time, data, constraint, filename,models)
        if time.time() - last_time >= batch_interval:
            print(time.time(), last_time, time.time() - last_time)
            last_time = time.time()
            for constraint in constraints.keys():
                overall_accuracy = constraints[constraint][2] / constraints[constraint][1] * 100
                step_accuracy = constraints[constraint][3] / constraints[constraint][1] * 100
                constraints[constraint][3]=0
                vote_based_scaling(step_accuracy,overall_accuracy,processor.correct_predictions[constraint],constraints[constraint][0], constraint)
                processor.correct_predictions[constraint] = defaultdict(list)
        logging.info(f'Processed request for model: {model_name} {name} {synset}  {filename} {models}')
        check_ground_truth(synset,filename, constraint)
        if (typ > 3):
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999 + 0.001
        elif (handel_time > UPPER_LATENCY_BOUND):
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999 +  0.001
        else:
            scheduler.Scheduler.failed_rate = scheduler.Scheduler.failed_rate * 0.999

        if sch.failed_rate > SLA_BOUND:
            #sch.launch_standby('c5.xlarge', 1, model_name)
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
