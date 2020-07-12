#coding: utf-8


UPPER_LATENCY_BOUND = 200
SLA_BOUND = 0.02

HANDLE_SIZE = 8

#batch size config
HANDLE_SIZE_P2 = 8
HANDLE_SIZE_C5=4
HANDLE_SIZE_C5X = 6
HANDLE_SIZE_C52X = 8
HANDLE_SIZE_C54X = 12

#proactive scheduler update interval
UPDATER_INTERVAL = 600

SERVING_PORT = 7002
API_PORT = 8301

API_TIMEOUT = 10.0

# use weighted load balancer
DEFAULT_BALANCER = 'weight'

# MangoDB is used to manage instance info
DB_HOST = 'localhost'
# REDIS_PORT = 6379
MONGO_PORT = 27017

#DB configs
ON_DEMAND_PRIZE_DB = 'on_demand'
SPOT_PRIZE_DB = 'spot'
AWS_DB = 'aws'
PRE_AWS_DB = 'pre_aws'
INS_DB = 'instance'
BACKUP_DB = 'back_up'
DEMAND_AWS_DB = 'demand_aws'
PRE_DEMAND_AWS_DB = 'pre_demand_aws'

SPOT_PRIZE_URL = 'http://spot-price.s3.amazonaws.com/spot.js'
ON_DEMAND_PRIZE_URL = 'http://a0.awsstatic.com/pricing/1/ec2/linux-od.js'

# the data amount to warm up predictor, the re-schedule period(seconds)
PREDICTOR_PARAM = [2, 30]

# the arrival rate smaple window
PREDICTOR_WINDOW = 5

SECURITY_GROUPS = {
    'us-west-2': ['sg-xxx'],
    'us-west-1': ['sg-xxx'],
    'us-east-1': ['sg-8c34ddfc'],
    'eu-west-3': ['sg-6c538705']
}
KEYS = {
    'us-east-1': 'aws-cocktail',
    'eu-west-3': 'aws-cocktail',
    'us-west-1': 'xxx',
    'us-west-2': 'xxx'
}

# which market to use
INS_SOURCE = 'spot'

# which model to serve
MODEL = 'tf'

# AMIs for each model
AMIS_TF = {
    'us-east-1': {'CPU': 'ami-0d97a358db413f1e8', 'GPU': 'ami-0edeb7664ef2967e7'},
    'eu-west-3': {'CPU': 'ami-0d97a358db413f1e8', 'GPU': 'ami-0edeb7664ef2967e7'}
}
AMIS_KR = {
    'us-east-1': {'CPU': 'ami-xxx', 'GPU': 'ami-xxx'}
}
AMIS_MX = {
    'us-east-1': {'CPU': 'ami-xxx'}
}
AMIS_NMT = {
    'us-east-1': {'CPU': 'ami-xxx', 'GPU': 'ami-xxx'}
}

AMIS_MODEL = {
    'kr': AMIS_KR,
    'tf': AMIS_TF,
    'mx': AMIS_MX,
    'nmt': AMIS_NMT
}

AMIS = AMIS_MODEL[MODEL]

# the instances to use in each model
AllIndexType = {
    'tf': ['c5.large', 'c5.xlarge', 'c5.2xlarge', 'c5.4xlarge'],
    'kr': ['c5.xlarge', 'c5.2xlarge', 'c5.4xlarge', 'p2.xlarge'],
    'mx': ['c5.xlarge', 'c5.2xlarge', 'c5.4xlarge', 'p2.xlarge'],
    'nmt': ['c5.xlarge', 'c5.2xlarge', 'c5.4xlarge', 'p2.xlarge']
}

# the profiled capacity for all instances (r/m)
AllCapacity = {
    'tf': [284, 490, 771, 1080, 415 * HANDLE_SIZE ],
    'kr': [85, 135, 187, 95 * HANDLE_SIZE ],
    'mx': [3100, 1, 1, 1],
    'nmt': [40 * HANDLE_SIZE_C5X, 54 * HANDLE_SIZE_C52X, 75 * HANDLE_SIZE_C54X, 80 * HANDLE_SIZE_P2]
}

# load balancer weight
AllWeight = {
    'tf': [284, 490, 771, 1080, 415],
    'kr': [85, 135, 187, 95],
    'mx': [3100, 1, 1, 1],
    'nmt': [40, 54, 75, 80]
}

IndexType = AllIndexType[MODEL]
Capacity = AllCapacity[MODEL]
Weights = AllWeight[MODEL]

Instance_Weights = {}
[ Instance_Weights.update({t: w}) for t, w in zip(IndexType, Weights) ]

# AWS credentials
filepath='/home/cc/aws-keys'
with open(filepath) as fp:
   access_key = fp.readline().strip()
   secret_key = fp.readline().strip()

DEFAULT_REGION = 'eu-west-3'
CREDENTIALS = {
    'aws_access_key_id' : access_key,
    'aws_secret_access_key' : secret_key
}
models="MobileNet ResNet50"

# model deploy cmd
TF_DEPLOY_CMD ={
    'cpu': f'nohup sudo docker run -p 8501:8501 --name TFserving_resnet --mount type=bind,source=/home/ubuntu/resnet,target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving  > server.log 2>&1 &',
    'CPU': f'nohup sudo python server.py {models} 50055 &  > server.log 2>&1 &',
    'GPU': f'nohup sudo docker run -p 8501:8501 --runtime=nvidia --name tfserving_resnet --mount type=bind,source=/home/ubuntu/resnet,target=/models/resnet -e MODEL_NAME=resnet -t tensorflow/serving:latest-gpu > server.log 2>&1 &',
    }


ITEM_DELIMITER = ','
