import argparse
import os
from pathlib import Path

import mmcv
from mmcv import Config

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, integrate
import pandas as pd

from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=999,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg


def parse_gt_bboxes(bboxes):
    box_ratios = []
    box_areas = []
    for bbox in bboxes:
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        box_ratios.append(round(w / h, 2))
        box_areas.append(int(w * h))
    return box_ratios, box_areas


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    img_ratios = []
    box_ratios = []
    box_areas = []
    box_nums = []
    labels = []
    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        # mmcv.imshow_det_bboxes(
        #     item['img'],
        #     item['gt_bboxes'],
        #     item['gt_labels'],
        #     class_names=dataset.CLASSES,
        #     show=not args.not_show,
        #     out_file=filename,
        #     wait_time=args.show_interval)
        img = item['img']
        height, width = img.shape[:2]
        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']

        # image ratio
        img_ratios.append(width / height)

        # boxes info
        ratios, areas = parse_gt_bboxes(gt_bboxes)
        box_ratios.extend(ratios)
        box_areas.extend(areas)
        box_nums.append(len(areas))

        # labels
        labels.append(gt_labels)

        progress_bar.update()
    img_ratios = np.array(img_ratios)
    box_ratios = np.array(box_ratios).clip(max=30)
    box_areas = np.array(box_areas).clip(max=3e4)
    box_nums = np.array(box_nums)
    labels = np.concatenate(labels)
    print(img_ratios)
    print(box_ratios)
    print(box_areas)
    print(box_nums)
    print(labels)
    img_ratios = pd.Series(img_ratios, name="img_ratios")
    box_ratios = pd.Series(box_ratios, name="box_ratios")
    box_areas = pd.Series(box_areas, name="box_areas")
    box_nums = pd.Series(box_nums, name="box_nums")
    labels = pd.Series(labels, name="labels")
    img_ratios = sns.kdeplot(img_ratios, color="b").get_figure()
    img_ratios.savefig(os.path.join(args.output_dir, 'img_ratios.jpg'), dpi=400)
    plt.cla()
    box_ratios = sns.kdeplot(box_ratios, color="b").get_figure()
    box_ratios.savefig(os.path.join(args.output_dir, 'box_ratios.jpg'), dpi=400)
    plt.cla()
    box_areas = sns.kdeplot(box_areas, color="b").get_figure()
    box_areas.savefig(os.path.join(args.output_dir, 'box_areas.jpg'), dpi=400)
    plt.cla()
    box_nums = sns.kdeplot(box_nums, color="b").get_figure()
    box_nums.savefig(os.path.join(args.output_dir, 'box_nums.jpg'), dpi=400)
    plt.cla()
    labels = sns.kdeplot(labels, color="b").get_figure()
    labels.savefig(os.path.join(args.output_dir, 'labels.jpg'), dpi=400)


if __name__ == '__main__':
    main()
