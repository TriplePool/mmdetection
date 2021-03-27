import os
import os.path as osp

import mmcv
from tqdm import tqdm


def parse_page_path(page_path):
    filename = osp.basename(page_path)
    filename = filename.split('.')[0]
    doc_id, page_id = filename.split('-')
    return doc_id, page_id


def parse_ann_file(ann_path):
    if not osp.exists(ann_path): return []
    with open(ann_path, encoding='utf-8') as f:
        res = []
        lines = f.readlines()
        for line in lines:
            if "#" in line:
                continue
            x, y, w, h, c = line[2:-1].split('\t ')
            box = Box(float(x), float(y), float(w), float(h), int(c))
            res.append(box)
        return res


class Page:
    def __init__(self, img_path, simple_prefix=False):
        if simple_prefix:
            self.ann_path = osp.basename(img_path).split('.')[0] + '.txt'
        else:
            self.doc_id, self.page_id = parse_page_path(img_path)
            self.ann_path = '{}-color_{}.txt'.format(self.doc_id, self.page_id)
        self.ann_path = osp.join(osp.dirname(img_path), self.ann_path)
        self.boxes = parse_ann_file(self.ann_path)


class Box:
    def __init__(self, x, y, w, h, class_id):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.class_id = class_id

    def rescale(self, x, y):
        self.x *= x
        self.y *= y
        self.w *= x
        self.h *= y


class OCR:
    def __init__(self, img_list_path, rec_path):
        self.data_dict = dict()
        self.vocab_dict = dict()

        with open(img_list_path) as f_img:
            img_lines = f_img.readlines()
        with open(rec_path) as f_rec:
            rec_lines = f_rec.readlines()
        print(len(img_lines), len(rec_lines))
        assert len(img_lines) == len(rec_lines)
        for i in tqdm(range(len(img_lines))):
            patch_path = img_lines[i][:-1]
            rec_line = rec_lines[i][:-1]
            rec_res = rec_line.split(' ')
            img_fn = osp.basename(osp.dirname(patch_path))
            if img_fn not in self.data_dict:
                self.data_dict[img_fn] = []
            self.data_dict[img_fn].append(rec_res)
            for vocab in rec_res:
                if vocab not in self.vocab_dict:
                    self.vocab_dict[vocab] = 0
                self.vocab_dict[vocab] += 1

    def get_rec_res(self, img_fn, obj_id):
        return self.data_dict[img_fn][obj_id]


def convert_icdar2021_to_coco(data_dir, out_file, image_prefix, with_sentence=False, simple_format=False,
                              without_ann=False):
    ocr = None
    if with_sentence:
        ocr = OCR('/home/wbl/workspace/data/ICDAR2021/{}_patches/imagelist.txt'.format(osp.basename(data_dir)),
                  '/home/wbl/workspace/data/ICDAR2021/{}_patches/pred.txt'.format(osp.basename(data_dir)))
    files = os.listdir(data_dir)
    img_files = []
    for f in files:
        if f.endswith(image_prefix):
            img_files.append(f)
    annotations = []
    images = []
    obj_count = 0
    progress_bar = mmcv.ProgressBar(len(img_files))
    for idx, v in enumerate(img_files):
        img_path = osp.join(data_dir, v)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=v,
            height=height,
            width=width))
        if without_ann:
            progress_bar.update()
            continue

        page = Page(img_path, simple_format)
        box_count = 0

        for box in page.boxes:
            box.rescale(width * 0.01, height * 0.01)
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=box.class_id,
                bbox=[box.x, box.y, box.w, box.h],
                area=box.w * box.h,
                segmentation=[[box.x, box.y, box.x + box.w, box.y, box.x + box.w, box.y + box.h, box.x, box.y + box.h]],
                iscrowd=0)
            if with_sentence:
                data_anno['sentence'] = ocr.get_rec_res(v.replace(image_prefix, ''), box_count)
            annotations.append(data_anno)
            obj_count += 1
            box_count += 1
        progress_bar.update()

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'embedded'}, {'id': 1, 'name': 'isolated'}])
    mmcv.dump(coco_format_json, out_file)


def convert_icdar2021_to_coco_one_class(data_dir, out_file, image_prefix, keep_class={'id': 0, 'name': 'embedded'},
                                        with_sentence=False):
    ocr = None
    if with_sentence:
        ocr = OCR('/home/wbl/workspace/data/ICDAR2021/{}_patches/imagelist.txt'.format(osp.basename(data_dir)),
                  '/home/wbl/workspace/data/ICDAR2021/{}_patches/pred.txt'.format(osp.basename(data_dir)))

    files = os.listdir(data_dir)
    img_files = []
    for f in files:
        if f.endswith(image_prefix):
            img_files.append(f)
    annotations = []
    images = []
    obj_count = 0
    progress_bar = mmcv.ProgressBar(len(img_files))
    for idx, v in enumerate(img_files):
        img_path = osp.join(data_dir, v)
        height, width = mmcv.imread(img_path).shape[:2]

        images.append(dict(
            id=idx,
            file_name=v,
            height=height,
            width=width))

        page = Page(img_path)
        box_count = 0

        for box in page.boxes:
            box.rescale(width * 0.01, height * 0.01)
            if box.class_id != keep_class['id']:
                continue
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=box.class_id,
                bbox=[box.x, box.y, box.w, box.h],
                area=box.w * box.h,
                segmentation=[[box.x, box.y, box.x + box.w, box.y, box.x + box.w, box.y + box.h, box.x, box.y + box.h]],
                iscrowd=0)
            if with_sentence:
                data_anno['sentence'] = ocr.get_rec_res(v.replace(image_prefix, ''), box_count)
            annotations.append(data_anno)
            obj_count += 1
            box_count += 1
        progress_bar.update()

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': 0, 'name': 'embedded'}, {'id': 1, 'name': 'isolated'}])
    mmcv.dump(coco_format_json, out_file)


if __name__ == '__main__':
    # convert_icdar2021_to_coco('/home/wbl/workspace/data/ICDAR2021/TrM',
    #                           '/home/wbl/workspace/data/ICDAR2021/TrM_with_sentence.json', '.jpg', with_sentence=True)
    # convert_icdar2021_to_coco_one_class('/home/wbl/workspace/data/ICDAR2021/TrM',
    #                                     '/home/wbl/workspace/data/ICDAR2021/TrM_embedded_with_sentence.json', '.jpg',
    #                                     with_sentence=True)
    # convert_icdar2021_to_coco_one_class('/home/wbl/workspace/data/ICDAR2021/TrM',
    #                                     '/home/wbl/workspace/data/ICDAR2021/TrM_isolated_with_sentence.json', '.jpg',
    #                                     {'id': 1, 'name': 'isolated'}, with_sentence=True)

    convert_icdar2021_to_coco('/home/wbl/workspace/data/ICDAR2021/test',
                              '/home/wbl/workspace/data/ICDAR2021/test.json', '.jpg', with_sentence=False,
                              simple_format=True, without_ann=True)
