import os
import os.path as osp

import mmcv
from tqdm import tqdm


def parse_page_path(page_path):
    filename = osp.basename(page_path)
    filename = filename.split('.')[0]
    doc_id, page_id = filename.split('-')
    return doc_id, page_id


def get_labels(ann_dir):
    res = set()
    file_list = os.listdir(ann_dir)
    for fn in file_list:
        if not str(fn).endswith('.txt'): continue
        fp = os.path.join(ann_dir, fn)
        boxes = parse_ann_file(fp)
        for box in boxes:
            res.add(box.label)
    return res


# {'table', 'figure', 'none', 'formula', 'textline'}

LABEL2ID = {'textline': 0, 'figure': 1, 'table': 2, 'formula': 3, 'none': 4}


def parse_ann_file(ann_path):
    if not osp.exists(ann_path): return []
    with open(ann_path, encoding='utf-8') as f:
        res = []
        lines = f.readlines()
        for line in lines:
            box, label = line[:-1].split('\t')
            # x1, y1, x2, y2 = box.split(',')
            # box = Box(float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1), label)
            # x1, y1, x2, y2 = box.split(',')
            # print(x1, y1, x2, y2)
            x1, x2, y1, y2 = box.split(',')
            box = Box(float(x1), float(y1), float(x2) - float(x1), float(y2) - float(y1), label)
            # box = Box(float(x1), float(y1), 20, 20, label)
            res.append(box)
        return res


class Box:
    def __init__(self, x, y, w, h, label):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.label = label

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


def convert_math_formula_to_coco(data_dir, out_file, without_ann=False):
    files = os.listdir(data_dir)
    img_files = []
    for f in files:
        if f.endswith('.bmp'):
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
            file_name=img_path,
            height=height,
            width=width))
        if without_ann:
            progress_bar.update()
            continue
        box_count = 0

        ann_fp = img_path.replace('.bmp', '.txt')
        ann_fp = ann_fp.replace('/img', '/anno')
        boxes = parse_ann_file(ann_fp)

        for box in boxes:
            data_anno = dict(
                image_id=idx,
                id=obj_count,
                category_id=LABEL2ID[box.label],
                bbox=[box.x, box.y, box.w, box.h],
                area=box.w * box.h,
                segmentation=[[box.x, box.y, box.x + box.w, box.y, box.x + box.w, box.y + box.h, box.x, box.y + box.h]],
                iscrowd=0)
            annotations.append(data_anno)
            obj_count += 1
            box_count += 1
        progress_bar.update()

    coco_format_json = dict(
        images=images,
        annotations=annotations,
        categories=[{'id': LABEL2ID[x], 'name': x} for x in LABEL2ID])
    mmcv.dump(coco_format_json, out_file)


if __name__ == '__main__':
    a_dir = '/home/wbl/workspace/data/MathFormula/train/anno'
    img_dir = '/home/wbl/workspace/data/MathFormula/train/img'
    opt_path = '/home/wbl/workspace/data/MathFormula/train.json'
    convert_math_formula_to_coco(img_dir, opt_path)