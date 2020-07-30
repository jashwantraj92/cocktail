import asyncio
import json
import logging
import threading
import time
from collections import deque
import numpy as np
import requests

import aiohttp
import time
import tensorflow as tf

from . import aws_manager, utils, naiveSchedule
from .model_source import mdl_source
from .load_balancer import get_balancer
from .load_balancer import get_model_tracker
from .data_accessor import instance_accessor, demand_aws_accessor
from .constants import *
from .instance_source import ins_source

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

REQ_CPU = 0
REQ_GPU = 1
REQ_LAMBDA_CPU = 2
REQ_LAMBDA_GPU = 3
REQ_FAIL_CPU = 4
REQ_FAIL_GPU = 5

class QueryProcessor():

    def set_loop(self, loop_):
        self.loop = loop_
        self.query_queue = QueryQuene()
        self.balancer = get_balancer()
        # self.instances = utils.parse_instances(instance_accessor.get_all_instances())
        # instance_accessor.subscribe(self.update_instances)
        self.session = aiohttp.ClientSession(loop=self.loop)
        asyncio.ensure_future(self._manage_queue())
        self.model_tracker = get_model_tracker()

            
    async def send_query(self, name, times, data, constraints):

        #constraints = data.split(",")[1]
        #logging.info(f"constraint is {constraints}")
        models = await self.get_models(int(constraints))
        #models = models.split()
        futures = []
        for i in range(len(models)):
            future = asyncio.Future()
            logging.info(f"adding query to queue {models[i]} {times}" )
            await self.query_queue.put(future, name, times, data, models[i])
            futures.append(future)
            self.model_tracker[models[i]].append([time.time(),1])
        #await future
        return await self.ensemble_result(futures, models)
        #return future.result()
    
    async def get_requirements(self,constraint):
        logging.info(f'constraint are{accuracy[constraint],latency[constraint]}')
        return float(accuracy[constraint]),latency[constraint]
    
    async def get_models(self, constraints):
        global inst_list, current_latency, current_cost
        print('main invoked') 
        accuracy,latency = await self.get_requirements(constraints)
        models = naiveSchedule.select_models(latency,accuracy,2)
        logging.info(f"selected models are {models}")
        return models
        #return ["MobileNetV2","InceptionV3"]
      
    async def _manage_queue(self):
        while True:
            """
            batch requests for p2 instances, batch size = 16
            """

            logging.info(f"waiting for querys {self.query_queue.size()}\n")
            info = await self.query_queue.get()
            name = info[0][1]
            fu, times, data, model = [i[0] for i in info], [i[2] for i in info], [i[3] for i in info], info[0][4]
            #constraints = json.dumps({'data':f'{data[0]}'}).strip("\"}").split(",")[1]
            #logging.info(f"constraint is {constraints}")
            #models = await self.get_models(int(constraints))
            #models = await self.get_models(0)
            statements = []
            #for i in range(len(models)):
            alloc_info = ins_source.get_ins_alloc(name, model, self.balancer)
            logging.info(f'sending query to VM &&&&&&&&&&&&&: {alloc_info} {fu} {times} {model}')
            if alloc_info:
                ip, typ = alloc_info[0], alloc_info[1]
                if self.query_queue.size() > 1:
                    if typ.startswith('p2'):
                        handle_size = HANDLE_SIZE_P2
                    elif typ.startswith('c5.x'):
                        handle_size = HANDLE_SIZE_C5X
                    elif typ.startswith('c5.2x'):
                        handle_size = HANDLE_SIZE_C52X
                    elif typ.startswith('c5.4x'):
                        handle_size = HANDLE_SIZE_C54X
                    elif typ.startswith('c5.'):
                        handle_size = HANDLE_SIZE_C5

                    if model.startswith('Mobil'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_Mobilenet - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is  {model}, {handle_size} {self.query_queue.size()}')
                    elif model.startswith('InceptionRe'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_InceptionResnet - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is {model}, {handle_size} {self.query_queue.size()}')
                    elif model.startswith('Inception'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_Inception - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is {model}, {handle_size} {self.query_queue.size()}')
                    elif model.startswith('Resn'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_Resnet - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is {model}, {handle_size} {self.query_queue.size()}')
                    elif model.startswith('Xcepti'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_Xception - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is {model}, {handle_size} {self.query_queue.size()}')
                    elif model.startswith('NASNetMo'):
                        other_info = await self.query_queue.get(handle_size * HANDLE_SIZE_Nasnetmobile - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                        logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is  {model}, {handle_size} {self.query_queue.size()}')

                    #data =  data[0] + "," + models[i]
                #statements.append(self.loop.create_task(self._get_result(fu, name, times, data, ip)))
                for f in fu:
                    self.loop.create_task(self._get_result(f, name, times, data, ip))
            else:
                [ f.set_result(('No resources available', -1, utils.gap_time(t))) for f, t in zip(fu, times) ]
            #logging.info(f'**************** waiting for results ********************') 
            #self.loop.create_task(self.ensemble_result(statements))

  
    async def ensemble_result(self, statements, models): 
        predictions = []
        votearray=[]    
        voteclasses= []
        typ=0
        all_times=[]
        times=0
        #results,failed = await asyncio.wait(statements,return_when=asyncio.ALL_COMPLETED)
        for future in statements:
            await future
            result, typ, times = future.result()
            all_times.append(times)
            logging.info(f"Future results is {result} {result.split()[0]} {typ} {times}")
            #votearray.append(result.result().split()[0])
            votearray.append(result.split(" ")[0])
            #voteclasses.append(result.result().split()[1])
            voteclasses.append(result.split()[1])
            predictions.append(result)
        maxVoteLabel        =       max(set(votearray), key = votearray.count) 
        maxVoteClass        =       max(set(voteclasses), key = voteclasses.count)
        logging.info(f'**************** gather results are {maxVoteClass}, {maxVoteLabel} {all_times}********************')
        return str(maxVoteLabel) + "," + str(maxVoteClass), int(typ), np.mean(np.array(all_times)), models
    
    async def _get_result(self, futures, name, times, data, ip):
        results, req_type = await self._serve(name, data, ip)
        logging.info(f'predicted result is {results} {futures}')
        process_time =  float(results.split()[2])*1000
        [ f.set_result((r, typ, process_time)) for f, t, r, typ in zip(futures, times, [results], req_type) ]
        #return str(results)

    async def _serve(self, name, data, ip):

        is_gpu = len(data) > 2
        req_type = [REQ_GPU for _ in data] if is_gpu else [REQ_CPU for _ in data]
        req = mdl_source.get_request(data, ip)

        logging.info(f'Send request to ip: {ip}; batch_size:{len(data)};')
        #data = data[0].split(",")[0]
        data = json.dumps({'data':data[0]})
        
        logging.info(f'data is {len(data)}')
        #resp = await self.session.post(mdl_source.get_request(data, ip))
        #cmd= "python3.6 python-grpc-async-server-example/client.py 1 1 50055 0"
        #utils.check_command(utils.get_session(ip), cmd, debug=True)
        resp = await self.session.post(f'http://{ip}:8000/predict', data=data, headers={"Content-type": "application/json"})
        logging.info(f'the posted response is : {resp}')
        if resp.status == 200:
            r = await resp.json()
            return (mdl_source.collect_result(r), req_type)
        else:
            logging.info(f'Request rejected. ip: {ip}; status: {resp.status}')
            return ([ r for _ in data ], req_typ)
            async with self.session.get(mdl_source.get_lambda_req()) as res_lam:
                if res_lam.status == 200:
                        r = await res_lam.text()
                        req_typ = [REQ_LAMBDA_GPU for _ in data] if is_gpu else [REQ_LAMBDA_CPU for _ in data]
                        return ([ r for _ in data ], req_typ)
                else:
                        logging.info(f'Lambda rejected. status: {res_lam.status}')
                        req_typ = [REQ_FAIL_GPU for _ in data] if is_gpu else [REQ_FAIL_CPU for _ in data]
                        return ([ f'Error code : {res_lam.status}' for _ in data ], req_typ)

class QueryQuene():
    def __init__(self):
        self.queue = asyncio.Queue()
    
    async def put(self, fu, name, time, data, model):
        await self.queue.put((fu, name, time, data, model))

    async def get(self, num=1):
        items = []
        while num > 0:
            item = await self.queue.get()
            items.append(item)
            num -= 1

        return items

    def empty(self):
        return self.queue.empty()
    
    def size(self):
        return self.queue.qsize()
