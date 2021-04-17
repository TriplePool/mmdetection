from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion
import mmcv
import numpy as np

CLASS2ID = {'embedded': 0, 'isolated': 1}
ID2CLASS = {0: 'embedded', 1: 'isolated'}


def xywh_to_xyxy(x, y, w, h, norm=False, width=0, height=0):
    x0, y0, x1, y1 = x, y, x + w, y + h
    if norm and width > 0 and height > 0:
        x0 /= width
        x1 /= width
        y0 /= height
        y1 /= height
    if x1 > 1:
        print(x, y, w, h, width, height)
        return None
    return [x0, y0, x1, y1]


def xyxy_to_xywh(x0, y0, x1, y1, denorm=False, width=0, height=0):
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    if denorm and width > 0 and height > 0:
        x *= width
        y *= height
        w *= width
        h *= height
    return [x, y, w, h]


def json_to_lisdict(json_path, image_info_dict, score_threshold=0.85):
    results = mmcv.load(json_path)
    res_dict = dict()
    for res in results:
        img_id, bbox, score, category_id = res['image_id'], res['bbox'], res['score'], res['category_id']
        if score < score_threshold: continue
        width, height = image_info_dict[img_id]['width'], image_info_dict[img_id]['height']
        bbox = xywh_to_xyxy(bbox[0], bbox[1], bbox[2], bbox[3], norm=True, width=width, height=height)
        if bbox is None:
            print(json_path)
            print(image_info_dict[img_id])
            continue
        if img_id not in res_dict:
            res_dict[img_id] = dict(isolated=dict(boxes=[], scores=[], labels=[]),
                                    embedded=dict(boxes=[], scores=[], labels=[]))
        res_dict[img_id][ID2CLASS[int(category_id)]]['boxes'].append(bbox)
        res_dict[img_id][ID2CLASS[int(category_id)]]['scores'].append(score)
        res_dict[img_id][ID2CLASS[int(category_id)]]['labels'].append(category_id)
    return res_dict


def lisdict_to_json(lisdict, json_path, image_info_dict):
    opt_list = []
    for img_id in lisdict:
        for bbox, score, category_id in zip(lisdict[img_id]['boxes'], lisdict[img_id]['scores'],
                                            lisdict[img_id]['labels']):
            width, height = image_info_dict[img_id]['width'], image_info_dict[img_id]['height']
            bbox = xyxy_to_xywh(bbox[0], bbox[1], bbox[2], bbox[3], denorm=True, width=width, height=height)
            opt_list.append(dict(image_id=img_id, bbox=bbox, score=score, category_id=category_id))

    mmcv.dump(opt_list, json_path)


def ensemble_models(ipt_json_paths, opt_json_path, img_ann_path, emb_weights, iso_weights,
                    method='weighted_boxes_fusion', iou_thr=0.3,
                    skip_box_thr=0.0001,
                    sigma=0.1):
    img_info_dicts = mmcv.load(img_ann_path)['images']
    img_info_dict = dict()
    for img_info in img_info_dicts:
        img_info_dict[img_info['id']] = img_info

    res_dicts = []
    res_dict = dict()
    for json_path in ipt_json_paths:
        res_dicts.append(json_to_lisdict(json_path, img_info_dict))
    for img_id in res_dicts[0]:
        emb_boxes_list, iso_boxes_list = [], []
        emb_scores_list, iso_scores_list = [], []
        emb_labels_list, iso_labels_list = [], []
        for i in range(len(res_dicts)):
            if img_id not in res_dicts[i]:
                emb_boxes_list.append([])
                iso_boxes_list.append([])
                emb_scores_list.append([])
                iso_scores_list.append([])
                emb_labels_list.append([])
                iso_labels_list.append([])
            else:
                emb_boxes_list.append(res_dicts[i][img_id]['embedded']['boxes'])
                emb_scores_list.append(res_dicts[i][img_id]['embedded']['scores'])
                emb_labels_list.append(res_dicts[i][img_id]['embedded']['labels'])
                iso_boxes_list.append(res_dicts[i][img_id]['isolated']['boxes'])
                iso_scores_list.append(res_dicts[i][img_id]['isolated']['scores'])
                iso_labels_list.append(res_dicts[i][img_id]['isolated']['labels'])

        if method == 'nms':
            emb_boxes, emb_scores, emb_labels = nms(emb_boxes_list, emb_scores_list, emb_labels_list,
                                                    weights=emb_weights,
                                                    iou_thr=iou_thr)
            iso_boxes, iso_scores, iso_labels = nms(iso_boxes_list, iso_scores_list, iso_labels_list,
                                                    weights=iso_weights,
                                                    iou_thr=iou_thr)
        elif method == 'soft_nms':
            emb_boxes, emb_scores, emb_labels = soft_nms(emb_boxes_list, emb_scores_list, emb_labels_list,
                                                         weights=emb_weights,
                                                         iou_thr=iou_thr,
                                                         sigma=sigma, thresh=skip_box_thr)
            iso_boxes, iso_scores, iso_labels = soft_nms(iso_boxes_list, iso_scores_list, iso_labels_list,
                                                         weights=iso_weights,
                                                         iou_thr=iou_thr,
                                                         sigma=sigma, thresh=skip_box_thr)
        elif method == 'non_maximum_weighted':
            emb_boxes, emb_scores, emb_labels = non_maximum_weighted(emb_boxes_list, emb_scores_list, emb_labels_list,
                                                                     weights=emb_weights,
                                                                     iou_thr=iou_thr,
                                                                     skip_box_thr=skip_box_thr)
            iso_boxes, iso_scores, iso_labels = non_maximum_weighted(iso_boxes_list, iso_scores_list, iso_labels_list,
                                                                     weights=iso_weights,
                                                                     iou_thr=iou_thr,
                                                                     skip_box_thr=skip_box_thr)
        else:
            emb_boxes, emb_scores, emb_labels = weighted_boxes_fusion(emb_boxes_list, emb_scores_list, emb_labels_list,
                                                                      weights=emb_weights,
                                                                      iou_thr=iou_thr,
                                                                      skip_box_thr=skip_box_thr)
            iso_boxes, iso_scores, iso_labels = weighted_boxes_fusion(iso_boxes_list, iso_scores_list, iso_labels_list,
                                                                      weights=iso_weights,
                                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr)
        boxes = emb_boxes.tolist()
        scores = emb_scores.tolist()
        labels = emb_labels.tolist()
        boxes.extend(iso_boxes.tolist())
        scores.extend(iso_scores.tolist())
        labels.extend(iso_labels.tolist())
        res_dict[img_id] = dict(boxes=boxes, scores=scores, labels=labels)

    lisdict_to_json(res_dict, opt_json_path, img_info_dict)


if __name__ == '__main__':
    # # val
    # ipt_paths = [
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.18.bbox.json',  # 92.15
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.24.bbox.json',  # 92.12
    #     # # '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da_20000.24.bbox.json',  # 92.179
    #     # # # yx
    #     # '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri25.29.bbox.json',
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri25.bbox.json',
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri29.bbox.json',  # ensemble 0.253 92.52
    #     # # '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri25.24.bbox.json'
    #     # # # '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri25.14.bbox.json',
    #     # # # '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri29.20.bbox.json', # ensemble 0.205 92.51
    #
    #     # vfnet
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet.iso.c.bbox.json',
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet_isolated_da.20.bbox.json',
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet_isolated_da.24.bbox.json',
    #     '/home/wbl/workspace/codes/ICDAR2021/mmdetection/crpn.iso.bbox.json',
    #
    # ]
    # test
    ipt_paths = [
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.18.test.bbox.json',  # 92.15
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.24.test.bbox.json',  # 92.12
        # yx
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri25.test.bbox.json',
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tri29.test.bbox.json',  # ensemble 0.253 92.52

        # vfnet
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet.iso.test.c.bbox.json',
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet_isolated_da.20.test.bbox.json',
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/vfnet_isolated_da.24.test.bbox.json', # ensemble 0.0 92.89
        '/home/wbl/workspace/codes/ICDAR2021/mmdetection/crpn.iso.test.bbox.json',

    ]
    # weights = [1, 0.9, 1, 0.9]
    # # weights = [1, 0.9, 1, 0.9, 0.9]
    # iou_thr = 0.35
    # method = 'non_maximum_weighted'
    # img_info_path = '/home/wbl/workspace/data/ICDAR2021/test.json'
    # opt_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet_da.18.24.yx.25.29.test.bbox.nmw.json'
    # ensemble_models(ipt_paths, opt_path, img_info_path, weights=weights, method=method, iou_thr=iou_thr)

    # weights = [1, 0.9, 0.9, 1]
    # weights = [1, 0.9, 1, 0.9, 0.9]
    embedded_weights = [1, 0.9, 1, 0.9, 0, 0, 0, 0]
    isolated_weights = [0, 0, 0, 0, 1, 1, 1, 0]
    iou_thr = 0.35
    method = 'non_maximum_weighted'
    img_info_path = '/home/wbl/workspace/data/ICDAR2021/test.json'
    opt_path = '/home/wbl/workspace/codes/ICDAR2021/mmdetection/tridentnet.vfnet.nmw.test.json'
    ensemble_models(ipt_paths, opt_path, img_info_path, emb_weights=embedded_weights, iso_weights=isolated_weights,
                    method=method, iou_thr=iou_thr)
