import json 
import sys
import os
cmd = "aws lambda invoke --invocation-type RequestResponse --function-name lambda-resnet --region us-east-1 --payload \'{\"url\": \"https://s3.amazonaws.com/mxnet-tests/images/dog3.jpg\"}\' output_file"
import subprocess
output = subprocess.check_output(cmd, shell=True)
print(output)
