import asyncio
import csv
import logging
import operator
import time
from datetime import datetime
from functools import reduce

from . import (aws_manager, load_predictor, prize_request,
               proactive_controller2, reactive_controller, utils)
from .constants import *
from .data_accessor import (demand_aws_accessor, instance_accessor,
                            pre_aws_accessor, pre_demand_aws_accessor)
from .instance_source import ins_source
from . import instance_source
from .load_balancer import get_model_tracker
import random,math
logging.basicConfig(format = '%(asctime)s %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    filename = 'scheduler_test.log',
                    level=logging.INFO)

Tag = 0
Times = 1.2

class Scheduler():

    failed_rate = 0.0
    cool_down = 0

    def set_loop(self, loop_):
        self.loop = loop_
        logging.info('starting scheduler')
        self.predictor = load_predictor.Predictor(init_load=2964, 
            model_path='/home/cc/cocktail/resources/my_model_32.h5',
            scaler_path='/home/cc/cocktail/resources/my_scaler.save')
        self.warm_up_num = {}
        self.count = {}
        self.res_list = []
        self.loop.create_task(self.schedule())
        self.model_tracker = get_model_tracker()
        logging.info('started scheduler')
    def calculate_weights(self, model_tracker):
        weights = []
        Max = 0
        for model in models:
            logging.info(f'length s {len(model_tracker[model])} {Max}')
            if len(model_tracker[model]) > Max:
                Max = len(model_tracker[model])
        for model in models:
            weights.append(float(len(model_tracker[model])/ Max))      
        return weights
    def record_request(self, model_name):
        self.count[model_name] = (self.count[model_name] + 1) if model_name in self.count else 1
        if model_name not in self.warm_up_num:
            self.warm_up_num[model_name] = PREDICTOR_PARAM[0]
            logging.info(f'****** adding to warmup {self.count[model_name]}')

    def launch_standby(self, instance_type, count, model_name):
        if Scheduler.cool_down > 0:
            # Scheduler.cool_down -= 1
            return
        logging.info(f': Launch {count} {instance_type} instances for model: {model_name}')
        ami = AMIS[DEFAULT_REGION]['GPU'] if instance_type.startswith('p2') else AMIS[DEFAULT_REGION]['CPU']
        params = {'imageId':ami, 'instanceType':instance_type, 'targetCapacity': count, 'key_value':[('exp_round', Tag)] }
        ins_source.launch_ins(model_name, params,instance_source.models)
        Scheduler.cool_down = 5
            
    async def schedule(self):
        col_max_values = lambda l1, l2 : [ max(e1, e2) for e1, e2 in zip(l1, l2)]
        total_cost = 0

        controller = proactive_controller2.ProactiveController(instance_info=[])
        #controller = reactive_controller.ReactiveController(instance_info=[])

        max_count = 0;

        max_count_window = 0;

        while True:
            # await asyncio.sleep(PREDICTOR_PARAM[1])

            for i in range(0,PREDICTOR_PARAM[1]//PREDICTOR_WINDOW):
                await asyncio.sleep(PREDICTOR_WINDOW)
                for name in self.count.keys():
                    logging.info(f': Scheduler model count: {self.count[name]}')
                    if self.count[name] > max_count_window:
                        max_count_window = self.count[name]
                    self.count[name] = 0
                logging.info(f': Scheduler max_count_window: {max_count_window}')

            max_count_window *= PREDICTOR_PARAM[1]//PREDICTOR_WINDOW
            logging.info(f': Scheduler multiplied max_count_window: {max_count_window}')

            if Scheduler.cool_down > 0:
                Scheduler.cool_down -= 1

            for name in self.count.keys():
                if self.warm_up_num[name] > 0:
                    forecasts = self.predictor.predict(max_count_window * Times)
                    logging.info(f': Scheduler forecasts {forecasts} {instance_source.models}')                 
                    self.count[name] = 0
                    max_count_window = 0
                    self.warm_up_num[name] -= 1
                    continue
                logging.info(f': Updating prize_list')
                currentInstance, prize_list = ins_source.get_current_ins_and_prize(name, IndexType)
                logging.info(f': currentInstance {currentInstance} prize_list {prize_list}')
                if prize_list is None:
                    logging.info(f'Prize uavailable for {IndexType}')
                    self.count[name] = 0
                    max_count_window = 0
                    continue
                instanceInfo = []
                logging.info(f' Updated prize_list')
                for i in range(len(IndexType)):
                    instanceInfo.append([Capacity[i], prize_list[i], prize_list[i] * 180])

                # forecasts = self.predictor.predict(max_count_window * Times)

                logging.info(f': Updated forecasts {currentInstance}{instanceInfo}')

                if max_count < max_count_window:
                    max_count = max_count_window

                forecasts = [max_count * Times] * 50
                logging.info(f': forecasts {forecasts}{max_count}')                 


                # controller = proactive_controller2.ProactiveController(instance_info=instanceInfo)
                results, launch, destroy = controller.schedule(forecasts, currentInstance, instanceInfo)

                self.res_list.append((results, launch, destroy))
                des = []
                if len(self.res_list) >= 6:
                    des_list = reduce(col_max_values, [ res[0] for res in self.res_list ])
                    des = [ max(c - d, 0) for c, d in zip(currentInstance, des_list) ]
                    self.res_list.remove(self.res_list[0])
                    
                logging.info(f'Results: {results}; Current: {currentInstance}; Launch: {launch}; Destroy: {des}')                
                cost = 0
                for i in range(len(currentInstance)):
                    cost += instanceInfo[i][1] * 60 * currentInstance[i]
                weights = self.calculate_weights(self.model_tracker)
                logging.info(f'weoghts are {weights}')
                for i in range(len(launch)):
                    if launch[i] > 0:
                        logging.info(f'Launch {launch[i]} {IndexType[i]} instances for model: {name}')
                        ami = AMIS[DEFAULT_REGION]['GPU'] if IndexType[i].startswith('p2') else AMIS[DEFAULT_REGION]['CPU']
                        for j in range(len(models)):
                            model = models[j]
                            logging.info(f'Launch {launch[i]} for model: {model} numinstances {math.ceil(launch[i]*weights[j])}')
                            params = {'imageId':ami, 'instanceType':IndexType[i], 'targetCapacity': math.ceil(launch[i]*weights[j]), 'key_value':[('exp_round', Tag)] }
                            #model = random.choice(models)
                            ins_source.launch_ins(name, params,[models[j]])
                            cost += instanceInfo[i][2] * launch[i]

                for i in range(len(des)):
                    if des[i] > 0:
                        model = random.choice(models)
                        logging.info(f'Kill {des[i]} {IndexType[i]} instances for model: {name} {model}')
                        ins_source.kill_ins(name, DEFAULT_REGION, IndexType[i], des[i], model)

                total_cost += cost
                logging.info(f'count: {self.count[name]}; cost: {cost}; total_cost: {total_cost}')
                self.count[name] = 0
                max_count_window = 0
                # max_count = 0
