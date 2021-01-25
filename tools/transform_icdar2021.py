import cv2
import os

import mmcv
import numpy as np


def distance_transform(ipt_img_path, opt_img_path):
    img = cv2.imread(ipt_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)

    # Normalize
    r = cv2.normalize(r, r, 0, 1.0, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(g, g, 0, 1.0, cv2.NORM_MINMAX) * 255
    b = cv2.normalize(b, b, 0, 1.0, cv2.NORM_MINMAX) * 255

    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))

    target_file = os.path.join(opt_img_path)
    # print("Writing target file {}".format(target_file))
    cv2.imwrite(target_file, transformed_image)
    # print(transformed_image.shape)


def merge_transform(ipt_img_path, opt_img_path):
    img = cv2.imread(ipt_img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=5)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=5)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=5)

    # Normalize
    r = cv2.normalize(r, r, 0, 1.0, cv2.NORM_MINMAX) * 255
    g = cv2.normalize(g, g, 0, 1.0, cv2.NORM_MINMAX) * 255
    b = cv2.normalize(b, b, 0, 1.0, cv2.NORM_MINMAX) * 255

    # merge the transformed channels back to an image
    # Merge the channels
    dist = cv2.merge((b, g, r))
    dist = cv2.cvtColor(dist, cv2.COLOR_BGR2GRAY)
    dist = cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX) * 255

    transformed_image = dist.astype(img.dtype)
    res = cv2.merge((transformed_image, img, img))

    target_file = os.path.join(opt_img_path)
    # print(os.path.exists('/home/wbl/workspace/data/ICDAR2021/VaM_merge'))
    if not cv2.imwrite(target_file, res):
        print(ipt_img_path)


if __name__ == '__main__':

    ipt_dir = '/home/wbl/workspace/data/ICDAR2021/TrM'
    files = os.listdir(ipt_dir)
    opt_dir = '/home/wbl/workspace/data/ICDAR2021/TrM_merge'
    img_files = []
    for f in files:
        if f.endswith('.jpg'):
            img_files.append(f)

    progress_bar = mmcv.ProgressBar(len(img_files))
    for img_fn in img_files:
        ipt_path = os.path.join(ipt_dir, img_fn)
        opt_path = os.path.join(opt_dir, img_fn)
        merge_transform(ipt_path, opt_path)
        progress_bar.update()
