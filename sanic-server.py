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
from PIL import Image
#tf.enable_eager_execution()
from sanic import Sanic
from sanic.response import json
import time,random
votearray = []
voteclasses =[] 
maxVoteLabel=None
maxVoteClass=None
threads = []
pretrained_models = []
app = Sanic(__name__)
sem = None
models = []
pos			=	int(0.2*100);
arr			=	np.zeros(10000);
arr[:pos]	=	1;
np.random.shuffle(arr);
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
    global threads,pretrained_models,maVoteLabel,maxVoteClass,votearray,voteclasses,models,arr
    votearray=[]
    voteclasses= []

    if request.method == 'POST':
        receive_time = time.time()
        question = request.json['data']
        filename = request.json['file']
      
        print(f'Received request',receive_time, filename)
        data = question
        img = data
        #print(img,"***")
	#data = request.json['data']
        #file = tf.keras.utils.get_file(
        #    str(receive_time)+".jpg",data)
        
        #img = data.decode('base64')
        image='/home/cc/ensembling/sample.jpg'
        #with open(image, 'w') as another_open_file:
        #    another_open_file.write(img)
        #another_open_file.close()
        #print("imge is " ,image)
        #print("****************************************")
        new_image = Image.fromarray(np.array(img, dtype='uint8'))
        #img = tf.keras.preprocessing.image.load_img(image, target_size=[224, 224])
        x = tf.keras.preprocessing.image.img_to_array(new_image)
        x = tf.keras.applications.mobilenet.preprocess_input(np.array(img)[tf.newaxis,...])
        #model = int(question[1])
        fail = random.choice(arr)
        pretrained_model = pretrained_models
        if (fail):
          
            pretrained_model.remove(random.choice(pretrained_model))
        print("pretrained_model after shuffle are", pretrained_model)
        for i in range(len(pretrained_model)):
            #tid = threading.Thread(target=predict, args=(pretrained_models[i],x,models[i]))
            predict(pretrained_model[i],x,models[i])
            #threads.append(tid)
            #tid.start()
        #for thread in threads:
        #    thread.join()
        end_time = time.time()
        print(pretrained_models,threads)
        threads.clear() 
        maxVoteLabel	=	max(set(votearray), key = votearray.count)
        maxVoteClass	=	max(set(voteclasses), key = voteclasses.count)
        print("gather result ",votearray,voteclasses,maxVoteLabel,maxVoteClass)
        print("query respone time ",end_time, maxVoteLabel, maxVoteClass)
        return json({'image': maxVoteLabel, 'class': maxVoteClass, 'time': end_time - receive_time})

def predict(model,x,name):
    global maxVoteLabel, maxVoteClass,votearray,voteclasses
    receive_time = time.time()
    print(f'Thread Start {receive_time} {name}')
    result_before_save = model(x)
    print("Result before saving",
    tf.keras.applications.mobilenet.decode_predictions(
        result_before_save.numpy())[0][0])
    votearray.append(tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][1])
    voteclasses.append(tf.keras.applications.mobilenet.decode_predictions(result_before_save.numpy())[0][0][0])
    


async def compute(request):
    if request.method == 'POST':
        receive_time = time.time()
        datas = request.json['data'].split(',')
        print(f'Received request',receive_time,datas[0],datas[1])
    return json({'hello': 'world'})

models = sys.argv[4].split(",")
#model2=sys.argv[2]
_port = int(sys.argv[2])
_host = sys.argv[1]
_workers = int(sys.argv[3])
app.run(host=_host, port=_port,workers=_workers)
    

