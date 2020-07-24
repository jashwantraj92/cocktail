import aiohttp
import asyncio
import async_timeout
import requests,json,sys,time
from keras.preprocessing.image import load_img
import random 

import json
import time
from os.path import abspath, dirname, join
from base64 import b64encode, b64decode
import pandas as pd
import requests
import base64,os
from PIL import Image
import numpy as np


async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def main():

    host = sys.argv[1]
    port = sys.argv[2]
    images=[]
    for filename in os.listdir('/home/cc/ensembling/CYAN/val'):
        images.append(filename)
    for i in range(10):
        filename = str(random.choice(images))
        file = "/home/cc/ensembling/CYAN/val/" + str(filename)
        receive_time = time.time()
        #file = tf.keras.utils.get_file("grace_hopper.jpg","https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
        #dl_request = requests.get("/home/cc/ensembling/CYAN/val/ILSVRC2010_val_00024988.JPEG", stream=True)
        #dl_request.raise_for_status()
        #with open(file, mode='rb') as file:
        #    img = file.read()
        #img = load_img(file)
        #base64_bytes = b64encode(img)
        #base64_string = base64_bytes.decode('utf-8')
        image = Image.open(file)
        #data = {}
        #data['img'] = base64.b64encode(img) 
        #jpeg_bytes = b64encode(img).decode('utf-8')
        #    data = "{'http://ec2-35-174-5-243.compute-1.amazonaws.com:8000/predict', 'data': '{\"data\": \"https://tensorflow.org/images/blogs/serving/cat.jpg\"}', 'timeout': 2, 'headers': '{\"Content-type\": \"application/json\"}'}"
        ip = "ec2-35-174-5-243.compute-1.amazonaws.com"
        #data = json.dumps({'data':"https://tensorflow.org/images/blogs/serving/cat.jpg"})
        data = json.dumps({'data':np.array(image).tolist()})
        print(data)
        async with aiohttp.ClientSession() as session:
             resp = await session.post(url=f'http://{host}:8000/predict', data=data, headers={"Content-type": "application/json"})
             print(f'the posted response is : {resp}')
             if resp.status == 200:
                 r = await resp.json()
             print("response is ",filename, r, "end time is ", time.time() - receive_time)
    """requests.post(f'http://{host}:{port}/predict/',
	      headers={"Content-type": "application/json"},
               data=json.dumps({	
               'data':"\"https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg\""})
	)"""

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
