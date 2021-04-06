import cv2
import mmcv
import os
import numpy as np

from mmdet.core.visualization import imshow_det_bboxes

if __name__ == '__main__':
    coco_path = 'tri29.test.bbox.json'
    img_info_path = '/home/wbl/workspace/data/ICDAR2021/test.json'
    img_dir = '/home/wbl/workspace/data/ICDAR2021/test'

    opt_path = '/home/wbl/workspace/data/ICDAR2021/test.png'

    img_infos = mmcv.load(img_info_path)
    if isinstance(img_infos, dict):
        img_infos = img_infos['images']
    img_info_dict = dict()
    for img_info in img_infos:
        img_info_dict[img_info['id']] = img_info

    coco_results = mmcv.load(coco_path)
    if isinstance(coco_results, dict):
        coco_results = coco_results['annotations']

    img_id = 100
    img_fn = img_info_dict[img_id]['file_name']
    img_path = os.path.join(img_dir, img_fn)

    img = cv2.imread(img_path)
    print(img.shape)
    for res in coco_results:
        if res['image_id'] == img_id:
            img_id, bbox, score, category_id = res['image_id'], res['bbox'], res['score'], \
                                               res['category_id']

            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imwrite(opt_path, img)
