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


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description='image generator')
    parser.add_argument('gen_amount',type=int, help='number of images being generated')
    parser.add_argument(
        '--base-prob',
        type=float,
        default=0.1,
        help='probability that will be divided uniformly to all middle segments, '
             'while the rest part will be arranged corresponding to segments\' height')
    args = parser.parse_args()
    return args

args=parse_args()

path = "/home/zhouyuxuan/ICDAR2021/"
# path = "/Volumes/ROSELIA/ICDAR2021/"

logger = logger_config(log_path=path + 'gen/log.txt', logging_name='sherco')

CLASSES = ('embedded', 'isolated')

coco = COCO(path + "TrM.json")
cat_ids = coco.get_cat_ids(cat_names=CLASSES)
cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
img_ids = coco.get_img_ids()
img_info = coco.load_imgs(img_ids)

top_segs = []
bot_segs = []
mid_segs = []
mid_heights = []

for imgn in range(len(img_info)):
    img_id = img_info[imgn]['id']
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    ann_info = coco.load_anns(ann_ids)
    img = Image.open(path + "TrM/" + img_info[img_id]['file_name'])

    lines = [0, 2048]
    for annn in range(len(ann_info)):
        if ann_info[annn]['category_id'] == 1:
            lines.append(ann_info[annn]['bbox'][1])
            lines.append(ann_info[annn]['bbox'][1] + ann_info[annn]['bbox'][3])

    lines = sorted(lines)
    assert lines[-1] == 2048

    i=0
    while i<len(lines)-1:
        j=i+1
        while lines[j] - lines[i] <= 20:
            j=j+1
        if j>=len(lines):
            break
        annids = []
        for annn in range(len(ann_info)):
            bbox = ann_info[annn]['bbox']
            if bbox[1] >= lines[i] and bbox[1] + bbox[3] <= lines[j]:
                annids.append(ann_info[annn]['id'])

        segment = {'img_id': img_id,
                   'top': lines[i],
                   'bottom': lines[j],
                   'height': lines[j] - lines[i],
                   'ann_ids': annids}

        if lines[i] == 0:
            top_segs.append(segment)
        elif lines[j] == 2048:
            bot_segs.append(segment)
        else:
            mid_segs.append(segment)

        i=j+1

top_segs = SortedList(top_segs, key=lambda seg: seg['height'])
bot_segs = SortedList(bot_segs, key=lambda seg: seg['height'])
mid_segs = SortedList(mid_segs, key=lambda seg: seg['height'])
mid_tot=len(mid_segs)

mid_heights = np.array([seg['height'] for seg in mid_segs])
mid_weights = np.sqrt(mid_heights)
prefix_sum = np.array(mid_weights)
for i in range(1, mid_tot):
    prefix_sum[i] = prefix_sum[i - 1] + prefix_sum[i]


def cut_image(gen_image, segment, img_id, ann_id, current_height):
    image_info = img_info[segment['img_id']]
    ori_anns = coco.load_anns(segment['ann_ids'])
    anns=copy.deepcopy(ori_anns)
    # anns = coco.load_anns(segment['ann_ids'])
    for ann in anns:
        ann['image_id'] = img_id
        ann['id'] = ann_id
        delta = current_height - segment['top']
        ann['bbox'][1] += delta
        for segmentation in ann['segmentation']:
            for i in range(1, len(segmentation), 2):
                segmentation[i] += delta
        ann_id += 1

    image = Image.open(path + "TrM/" + image_info['file_name'])
    image = image.crop((0, segment['top'], 1447, segment['bottom']))
    # image = image.resize(image.size, Image.BILINEAR)
    gen_image.paste(image, box=(0, current_height, 1447, current_height + image.size[1]))

    return anns, segment['height']


img = Image.open(path + "TrM/" + img_info[0]['file_name'])
mode = img.mode

top_frq = np.zeros(len(top_segs))
bot_frq = np.zeros(len(bot_segs))
mid_frq = np.zeros(len(mid_segs))

TOT_GEN=args.gen_amount
BASE_PROB=args.base_prob

images = []
annotations = []
ann_id = 0
for gen_id in tqdm(range(TOT_GEN)):
    remain = 2048
    current_height = 0
    noceil = 0

    seg_id = random.randint(0,len(top_segs)-1)
    top_segment = top_segs[seg_id]
    # print(top_segment)
    remain -= top_segment['height']
    file_name = "top" + str(seg_id)
    top_frq[seg_id] += 1

    bot_segment = None
    if remain >= bot_segs[0]['height']:
        idx = bot_segs.bisect_left({'height': remain})
        seg_id = random.randint(0, idx)
        bot_segment = bot_segs[seg_id]
        remain -= bot_segment['height']
        file_name += "-bot" + str(seg_id)
        bot_frq[seg_id] += 1
    # print(bot_segment)

    gen_image = Image.new("L", (1447, 2048), color=255)

    annotation, height = cut_image(gen_image, top_segment, gen_id, ann_id, current_height)
    annotations.extend(annotation)
    current_height += math.ceil(height)
    # noceil+=height
    ann_id += len(annotation)

    if remain >= mid_segs[0]['height']:
        file_name += "-mid"
    while remain >= mid_segs[0]['height']:
        idx = min(mid_tot-1,mid_segs.bisect_left({'height': remain}))
        seg_id=np.random.choice(a=np.arange(idx+1),
                                p=BASE_PROB/(idx+1)+(1-BASE_PROB)*mid_weights[:idx+1]/prefix_sum[idx])
        mid_segment = mid_segs[seg_id]
        remain -= mid_segment['height']
        annotation, height = cut_image(gen_image, mid_segment, gen_id, ann_id, current_height)
        annotations.extend(annotation)
        current_height += math.ceil(height)
        ann_id += len(annotation)
        file_name += str(seg_id) + "-"
        mid_frq[seg_id] += 1
        # noceil+=height
        # print(ann_id,remain,current_height,noceil)

    if bot_segment != None:
        annotation, height = cut_image(gen_image, bot_segment, gen_id, ann_id, current_height)
        annotations.extend(annotation)
        current_height += math.ceil(height)
        # noceil+=height
        ann_id += len(annotation)

    file_name = file_name[:-1] + ".jpg"
    gen_image.save(path + "gen/" + file_name)

    images.append({"id": gen_id,
                   "file_name": file_name,
                   "height": 2048,
                   "width": 1447})

    # logger.info("image "+str(gen_id)+" generated")

json_file = open(path + "all.json", "w")
json_file.write(json.dumps({'images': images,
                            'annotations': annotations,
                            'categories': [{'id': 0, 'name': 'embedded'}, {'id': 1, 'name': 'isolated'}]}))
json_file.close()

top_frq.sort()
bot_frq.sort()


np.savetxt(path + "top_frq.txt", top_frq[::-1])
np.savetxt(path + "bot_frq.txt", bot_frq[::-1])

mid_rank=np.argsort(mid_frq)
f=open(path+"mid_frq.txt","w")
for i in range(mid_tot-1,-1,-1):
    f.write(str(mid_frq[mid_rank[i]])+" "+str(mid_rank[i])+" "+str(mid_segs[mid_rank[i]]['height'])+"\n")
f.close()

# coco = COCO(path + "gen.json")
# cat_ids = coco.get_cat_ids(cat_names=CLASSES)
# cat2label = {cat_id: i for i, cat_id in enumerate(cat_ids)}
# img_ids = coco.get_img_ids()
# img_info = coco.load_imgs(img_ids)
#
# for imgn in range(len(img_info)):
#     img_id = img_info[imgn]['id']
#     ann_ids = coco.get_ann_ids(img_ids=[img_id])
#     ann_info = coco.load_anns(ann_ids)
#     img = Image.open(path + "gen/" + img_info[img_id]['file_name'])
#
#     for annn in range(len(ann_info)):
#         # print(ann_info[annn])
#         bbox = ann_info[annn]['bbox']
#         # print(bbox,annn)
#         img2 = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
#         img2.save(path + "gen/crop" + str(annn) + ".jpg")
