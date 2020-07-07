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
import threading
import tensorflow as tf
import numpy as np

#tf.enable_eager_execution()
from sanic import Sanic
from sanic.response import json
import time
votearray = []
voteclasses =[] 
maxVoteLabel=None
maxVoteClass=None
threads = []
pretrained_models = []
app = Sanic(__name__)
sem = None
models = []
@app.listener('before_server_start')
def init(sanic, loop):
    global sem,models,pretrained_models,maxVoteLabel,maxVoteClass
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
        #print(pretrained_model)
        result_before_save = pretrained_model(x)
        print(pretrained_model, "Result before saving",
              tf.keras.applications.mobilenet.decode_predictions(
                 result_before_save.numpy())[0][0])
    #sem = asyncio.Semaphore(concurrency_per_worker, loop=loop)


@app.route('/predict',methods=['POST'])
 
async def test(request):
    global threads,pretrained_models,maxVoteLabel,maxVoteClass
    if request.method == 'POST':
        receive_time = time.time()
        print(f'Received request',receive_time)
        question = request.json['data'].split(',')
        data = question[0]
	#data = request.json['data']
        file = tf.keras.utils.get_file(
            str(receive_time)+".jpg",data)
        img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = tf.keras.applications.mobilenet.preprocess_input(np.array(img)[tf.newaxis,...])
        #model = int(question[1])
        for i in range(len(pretrained_models)):
            tid = threading.Thread(target=predict, args=(pretrained_models[i],x,))
            threads.append(tid)
            tid.start()
        for thread in threads:
            thread.join()
        end_time = time.time()
        print(pretrained_models,data,threads)
        threads.clear() 

        print("query respone time ",end_time, maxVoteLabel, maxVoteClass)
        return json({'image': maxVoteLabel, 'class': maxVoteClass})

def predict(model,x):
    votearray=[]
    voteclasses= []
    global maxVoteLabel, maxVoteClass
    receive_time = time.time()
    print(f'Thread Start {receive_time}')
    result_before_save = model(x)
    print("Result before saving",
    tf.keras.applications.mobilenet.decode_predictions(
        result_before_save.numpy())[0][0])
    votearray.append(tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][1])
    voteclasses.append(tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][0])
    maxVoteLabel	=	max(set(votearray), key = votearray.count)
    maxVoteClass	=	max(set(voteclasses), key = voteclasses.count)



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
    

