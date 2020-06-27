from __future__ import absolute_import, division, print_function
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl
from matplotlib import pyplot as plt
import grpc,sys
import time
import asyncio
from concurrent import futures
from threading import current_thread
from collections import defaultdict

import tensorflow as tf
import numpy as np


from sanic import Sanic
from sanic.response import json
import time
pretrained_models = []
app = Sanic(__name__)
sem = None
models = []
@app.listener('before_server_start')
def init(sanic, loop):
    global sem,models,pretrained_models
    print("listener function")
    #concurrency_per_worker = 4
    for model in models:
        pretrained_model = eval('tf.keras.applications.{0}()'.format(model))
        pretrained_models.append(pretrained_model)
    #pretrained_model = eval('tf.keras.applications.{0}()'.format(model))
    #pretrained_models.append(pretrained_model)
        file = tf.keras.utils.get_file(
        "cat.jpg",
        "https://tensorflow.org/images/blogs/serving/cat.jpg" )
        img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.mobilenet.preprocess_input(np.array(img)[tf.newaxis,...])
    #print("preductingfor url ",url, model)
        print(pretrained_model)
        result_before_save = pretrained_model(x)
        print("Result before saving",
              tf.keras.applications.mobilenet.decode_predictions(
                 result_before_save.numpy())[0][0])
    #sem = asyncio.Semaphore(concurrency_per_worker, loop=loop)


@app.route('/predict',methods=['POST'])
async def test(request):
    global pretrained_models 
    if request.method == 'POST':
        receive_time = time.time()
        print(f'Received request',receive_time)
        #question = request.json['data'].split(',')
        data = request.json['data'].split(',')
        file = tf.keras.utils.get_file(
            "grace_hopper.jpg",data)
        img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.mobilenet.preprocess_input(np.array(img)[tf.newaxis,...])
        #model = int(question[1])
        print(pretrained_models,data)
        for model in pretrained_models:
            result_before_save = model(x)
            print("Result before saving",
              tf.keras.applications.mobilenet.decode_predictions(
                  result_before_save.numpy())[0][0])
            end_time = time.time()
            print("query respone time ",end_time)
            results.append(result_before_save.numpy())[0][0])

    return json({'hello': 'world'})


async def compute(request):
    if request.method == 'POST':
        receive_time = time.time()
        datas = request.json['data'].split(',')
        print(f'Received request',receive_time,datas[0],datas[1])
    return json({'hello': 'world'})

models = sys.argv[4].split()
#model2=sys.argv[2]
_port = int(sys.argv[2])
_host = sys.argv[1]
_workers = int(sys.argv[3])
app.run(host=_host, port=_port,workers=_workers)
    

