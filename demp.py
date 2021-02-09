# coding: utf-8
import multiprocessing
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
import argparse

from modules.constants import *
from modules import data_accessor
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=SERVING_PORT)
    parser.add_argument('--signal', type=int, default=0)
    parser.add_argument('--need-updater', type=bool, default=False)
    parser.add_argument('--tag', type=int, default=0)
    return parser.parse_args()

def copy_keys(args, keys):
    dst_dict = {}
    src_dict = vars(args)
    for key in keys:
        if src_dict[key] is not None:
            dst_dict[key] = src_dict[key]
    return dst_dict

def main():
     data_accessor.init()
      
if __name__ == '__main__':
    main()
