import asyncio
import json
import logging
import threading
import time
from collections import deque

import requests

import aiohttp
import tensorflow as tf

from . import aws_manager, utils, naiveSchedule
from .model_source import mdl_source
from .load_balancer import get_balancer
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
        self.loop.create_task(self._manage_queue())
         
    async def send_query(self, name, time, data):
        future = asyncio.Future()
        logging.info("adding query to queue")
        await self.query_queue.put(future, name, time, data)
        await future
        return future.result()
    
    async def get_requirements(self,constraint):
        logging.info(f'accuracy[constraint],latency[constraint]')
        return float(accuracy[constraint]),latency[constraint]
    
    async def get_models(self, constraints):
        global inst_list, current_latency, current_cost
        print('main invoked') 
        """accuracy,latency = await self.get_requirements(constraints)
        models = naiveSchedule.select_models(latency,accuracy)
        logging.info(f"selected models are {models}")
        return models"""
        return ["MobileNetV2","InceptionV3"]
      
    async def _manage_queue(self):
        while True:
            """
            batch requests for p2 instances, batch size = 16
            """

            logging.info("waiting for querys\n")
            info = await self.query_queue.get()
            name = info[0][1]
            fu, times, data = [i[0] for i in info], [i[2] for i in info], [i[3] for i in info]
            #constraints = json.dumps({'data':f'{data[0]}'}).strip("\"}").split(",")[1]
            #logging.info(f"constraint is {constraints}")
            #models = await self.get_models(int(constraints))
            models = await self.get_models(0)
            statements = []
            for i in range(len(models)):
                alloc_info = ins_source.get_ins_alloc(name, models[i], self.balancer)
                logging.info(f'sending query to VM &&&&&&&&&&&&&: {alloc_info} {fu} {times}')
                if alloc_info:
                    ip, typ = alloc_info[0], alloc_info[1]
                    if typ.startswith('p2'):
                        other_info = await self.query_queue.get(HANDLE_SIZE_P2 - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                    elif typ.startswith('c5.x'):
                        other_info = await self.query_queue.get(HANDLE_SIZE_C5X - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                    elif typ.startswith('c5.2x'):
                        other_info = await self.query_queue.get(HANDLE_SIZE_C52X - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                    elif typ.startswith('c5.4x'):
                        other_info = await self.query_queue.get(HANDLE_SIZE_C54X - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]
                    elif typ.startswith('c5.'):
                        other_info = await self.query_queue.get(HANDLE_SIZE_C5 - 1)
                        [ (fu.append(i[0]), times.append(i[2]), data.append(i[3])) for i in other_info ]

                    #data =  data[0] + "," + models[i]
                    logging.info(f'candidate VM  is &&&&&&&&&&&&& {ip} data is {data} {other_info}')
                    statements.append(self.loop.create_task(self._get_result(fu, name, times, data, ip)))
                else:
                    [ f.set_result(('No resources available', -1, utils.gap_time(t))) for f, t in zip(fu, times) ]
            logging.info(f'**************** waiting for results ********************') 
            self.loop.create_task(self.ensemble_result(statements))

  
    async def ensemble_result(self, statements): 
        predictions = []
        votearray=[]    
        voteclasses= []
        results,failed = await asyncio.wait(statements,return_when=asyncio.ALL_COMPLETED)
        for result in results:
            votearray.append(result.result().split()[0])
            voteclasses.append(result.result().split()[1])
            predictions.append(result.result())
        maxVoteLabel        =       max(set(votearray), key = votearray.count) 
        maxVoteClass        =       max(set(voteclasses), key = voteclasses.count)
        logging.info(f'**************** gather results are {maxVoteClass}, {maxVoteLabel} ********************')
       
    async def _get_result(self, futures, name, times, data, ip):
        results, req_type = await self._serve(name, data, ip)
        logging.info(f'predicted result is {results}')
        #[ f.set_result((r, typ, utils.gap_time(t))) for f, t, r, typ in zip(futures, times, results, req_type) ]
        return str(results)

    async def _serve(self, name, data, ip):

        is_gpu = len(data) > 2
        req_type = [REQ_GPU for _ in data] if is_gpu else [REQ_CPU for _ in data]
        req = mdl_source.get_request(data, ip)

        logging.info(f'Send request to ip: {ip}; batch_size:{len(data)}; req {req}')
        data = json.dumps({'data':f'{data[0]}'})
        #resp = await self.session.post(mdl_source.get_request(data, ip))
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
    
    async def put(self, fu, name, time, data):
        await self.queue.put((fu, name, time, data))

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
