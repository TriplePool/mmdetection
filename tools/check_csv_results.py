import cv2
import csv
import pandas as pd
import mmcv
import os
import numpy as np
from tqdm import tqdm

from mmdet.core.visualization import imshow_det_bboxes


def load_predicted(path):
    """
    Loads the bounding boxes of csv file "path".
    Input:
    "path": csv file containing one bounding box per line.
    Output:
    "pages": dictionary with key = page_name, value = array containing
    the predicted bounding boxes
    """
    rows = pd.read_csv(path, header=None).to_numpy()
    pages = {}

    with tqdm(desc="Processing '" + path + "':", total=len(rows), ascii=True,
              bar_format='{desc} |{bar:10}| {percentage:3.0f}%') as pbar:
        for r in rows:
            # Remove the bounding boxes that have a confidence score of less than 0.05
            if r[-2] >= 0.05:
                page_name = r[0]
                list_BBs = pages.get(page_name, [])

                # [xmin, ymin, xmax, ymax, class_id, confidence]
                bb = [int(r[1]), int(r[2]), int(r[3]), int(r[4])]
                list_BBs.append(bb)
                pages[page_name] = list_BBs
                pbar.update(1)

    return pages


if __name__ == '__main__':
    csv_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet.vfnet.nmw.test.csv'
    img_dir = '/home/wbl/workspace/data/ICDAR2021/test'

    opt_path = '/home/wbl/workspace/data/ICDAR2021/test.png'

    img_fn = 'p0314.jpg'
    img_path = os.path.join(img_dir, img_fn)

    img = cv2.imread(img_path)
    print(img.shape)

    pages_res = load_predicted(csv_path)
    bboxs = pages_res[img_fn]

    for bbox in bboxs:
        x_min, y_min = int(bbox[0]), int(bbox[1])
        x_max, y_max = int(bbox[2]), int(bbox[3])

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite(opt_path, img)
