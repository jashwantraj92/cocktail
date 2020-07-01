import aiohttp
import asyncio
import async_timeout
import requests,json,sys,time
async def fetch(session, url):
    with async_timeout.timeout(10):
        async with session.get(url) as response:
            return await response.text()

async def main():

    receive_time = time.time()
    host = sys.argv[1]
    port = sys.argv[2]
    data = "{'http://ec2-35-174-5-243.compute-1.amazonaws.com:8000/predict', 'data': '{\"data\": \"https://tensorflow.org/images/blogs/serving/cat.jpg\"}', 'timeout': 2, 'headers': '{\"Content-type\": \"application/json\"}'}"
    ip = "ec2-35-174-5-243.compute-1.amazonaws.com"
    data = json.dumps({'data':"https://tensorflow.org/images/blogs/serving/cat.jpg"})
    print(data)
    async with aiohttp.ClientSession() as session:
        resp = await session.post(url=f'http://{host}:8000/predict', data=data, headers={"Content-type": "application/json"})
        print(f'the posted response is : {resp}')
        if resp.status == 200:
           r = await resp.json()
           print("response is ",r, "end time is ", time.time() - receive_time)
    """requests.post(f'http://{host}:{port}/predict/',
	      headers={"Content-type": "application/json"},
               data=json.dumps({	
               'data':"\"https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg\""})
	)"""

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
