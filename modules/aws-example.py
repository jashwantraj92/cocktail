import json
import logging
import operator
import tempfile
import time
from datetime import datetime

import boto3
import requests
from celery import Celery, task


CREDENTIALS = {
    'aws_access_key_id' : 'AKIA3745J3PL3FATPEH4',
    'aws_secret_access_key' : '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'
}


def get_client(region='us-east-1'):
    return boto3.client('ec2', region_name=region, **CREDENTIALS)

params={'imageId':'ami-04fe13c834f9bc166', 'instanceType':'c5.large', 'targetCapacity':1, 'key_value':[('exp_round', 0)] }

base = {
        'TargetCapacity': params['targetCapacity'],
        'TerminateInstancesWithExpiration': True,
        'ValidFrom': datetime(2020, 5, 10 ),
        'ValidUntil': datetime(2020, 10, 10),
        'IamFleetRole': 'arn:aws:iam::824426748887:role/aws-ec2-spot-fleet-tagging-role',
        'LaunchSpecifications': [{
            'ImageId': params['imageId'],
            'KeyName': 'aws-cocktail.pem',
            'InstanceType': params['instanceType'],
            'BlockDeviceMappings': [{
                'VirtualName': 'Root',
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 75,
                    'VolumeType': 'gp2',
                    'DeleteOnTermination': True
                }
            }],
            'Monitoring': {
                'Enabled': False
            }
        }],
        'AllocationStrategy': 'lowestPrice',
        'Type': 'maintain'
    }
  
client = get_client()
res = client.request_spot_fleet(

        SpotFleetRequestConfig=base
    )
info = aws_accessor.get_cluster(name)
    instances = []
    for _, id in info['info'].items():
        ec2 = boto3.resource('ec2', region_name=id['region'], **CREDENTIALS)
        [ instances.append(ec2.Instance(i)) for i in id['instance_id_list'] ]
    return instances
