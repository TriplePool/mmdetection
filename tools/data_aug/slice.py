from pycocotools.coco import COCO
from PIL import Image
from sortedcontainers import SortedList
import random
import json
import math
from tqdm import tqdm
import logging
import copy
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='data slicer')
    parser.add_argument('slice_amount',type=int, help='number of images needed')
    args = parser.parse_args()
    return args

args=parse_args()

path = "/home/zhouyuxuan/ICDAR2021/"

CLASSES = ('embedded', 'isolated')

n=args.slice_amount

coco = COCO(path + "all.json")
cat_ids = coco.get_cat_ids(cat_names=CLASSES)
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
img_ids = coco.get_img_ids()
images = coco.load_imgs(img_ids[:n])
ann_ids = coco.get_ann_ids(img_ids=img_ids[:n])
annotations = coco.load_anns(ann_ids)


json_file = open(path + "gen.json", "w")
json_file.write(json.dumps({'images': images,
                            'annotations': annotations,
                            'categories': [{'id': 0, 'name': 'embedded'}, {'id': 1, 'name': 'isolated'}]}))
json_file.close()
