import csv
import mmcv


def xywh_to_xyxy(x, y, w, h, norm=False, width=0, height=0):
    x0, y0, x1, y1 = x, y, x + w, y + h
    if norm and width > 0 and height > 0:
        x0 /= width
        x1 /= width
        y0 /= height
        y1 /= height
    return [x0, y0, x1, y1]


def convert_coco_to_icdar2021(coco_result_path, image_info_path, output_path, threshold=0.5, reformat=False):
    img_infos = mmcv.load(image_info_path)
    if isinstance(img_infos, dict):
        img_infos = img_infos['images']
    img_info_dict = dict()
    for img_info in img_infos:
        img_info_dict[img_info['id']] = img_info

    coco_results = mmcv.load(coco_result_path)
    if isinstance(coco_results, dict):
        coco_results = coco_results['annotations']
    csv_rows = []
    # print(len(coco_results))
    for coco_result in coco_results:
        img_id, bbox, score, category_id = coco_result['image_id'], coco_result['bbox'], coco_result['score'], \
                                           coco_result['category_id']
        print(score)
        if score < threshold:
            continue
        bbox = xywh_to_xyxy(bbox[0], bbox[1], bbox[2], bbox[3])
        img_fn = img_info_dict[img_id]['file_name']
        if reformat:
            csv_row = [img_fn, bbox[0], bbox[1], bbox[2], bbox[3], 1.0, int(category_id)]
        else:
            csv_row = [img_fn, bbox[0], bbox[1], bbox[2], bbox[3], score, int(category_id)]
        csv_rows.append(csv_row)

    with open(output_path, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(csv_rows)


if __name__ == '__main__':
    # coco_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.18.24.yx.25.29.test.bbox.nmw.json'
    # img_info_path = '/home/wbl/workspace/data/ICDAR2021/test.json'
    # opt_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.18.24.yx.25.29.test.bbox.nmw.csv'
    #
    # convert_coco_to_icdar2021(coco_path, img_info_path, opt_path, threshold=0.253, reformat=True)

    coco_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da_20000.ensemble.bbox.nmw.json'
    img_info_path = '/home/wbl/workspace/data/ICDAR2021/VaM.json'
    opt_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da_20000.ensemble.bbox.nmw.csv'

    convert_coco_to_icdar2021(coco_path, img_info_path, opt_path, threshold=0.206, reformat=True)
