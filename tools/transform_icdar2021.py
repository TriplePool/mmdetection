import cv2
import os

import mmcv


def distance_transform(ipt_img_path, opt_img_path):
    img = cv2.imread(ipt_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform transformations on image
    b = cv2.distanceTransform(img, distanceType=cv2.DIST_L2, maskSize=3)
    g = cv2.distanceTransform(img, distanceType=cv2.DIST_L1, maskSize=3)
    r = cv2.distanceTransform(img, distanceType=cv2.DIST_C, maskSize=3)

    # merge the transformed channels back to an image
    transformed_image = cv2.merge((b, g, r))
    target_file = os.path.join(opt_img_path)
    # print("Writing target file {}".format(target_file))
    cv2.imwrite(target_file, transformed_image)
    # print(transformed_image.shape)


if __name__ == '__main__':

    ipt_dir = '/home/wbl/workspace/data/ICDAR2021/VaM'
    files = os.listdir(ipt_dir)
    opt_dir = '/home/wbl/workspace/data/ICDAR2021/VaM_dis'
    img_files = []
    for f in files:
        if f.endswith('.jpg'):
            img_files.append(f)

    progress_bar = mmcv.ProgressBar(len(img_files))
    for img_fn in img_files:
        ipt_path = os.path.join(ipt_dir, img_fn)
        opt_path = os.path.join(opt_dir, img_fn)
        distance_transform(ipt_path, opt_path)
        progress_bar.update()
