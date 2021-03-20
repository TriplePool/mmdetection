from ensemble_boxes import nms, soft_nms, non_maximum_weighted, weighted_boxes_fusion
import mmcv


def xywh_to_xyxy(x, y, w, h, norm=False, width=0, height=0):
    x0, y0, x1, y1 = x, y, x + w, x + h
    if norm and width > 0 and height > 0:
        x0 /= width
        x1 /= width
        y0 /= height
        y1 /= height
    return [x0, y0, x1, y1]


def xyxy_to_xywh(x0, y0, x1, y1, denorm=False, width=0, height=0):
    x, y, w, h = x0, y0, x1 - x0, y1 - y0
    if denorm and width > 0 and height > 0:
        x *= width
        y *= width
        w *= height
        h *= height
    return [x, y, w, h]


def json_to_lisdict(json_path, image_info_dict):
    results = mmcv.load(json_path)
    res_dict = dict()
    for res in results:
        img_id, bbox, score, category_id = res['image_id'], res['bbox'], res['score'], res['category_id']
        width, height = image_info_dict[img_id]['width'], image_info_dict[img_id]['height']
        bbox = xywh_to_xyxy(bbox[0], bbox[1], bbox[2], bbox[3], norm=True, width=width, height=height)
        if img_id not in res_dict:
            res_dict[img_id] = dict(boxes=[], scores=[], labels=[])
        res_dict[img_id]['boxes'].append(bbox)
        res_dict[img_id]['scores'].append(score)
        res_dict[img_id]['labels'].append(category_id)
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


def ensemble_models(ipt_json_paths, opt_json_path, img_ann_path, weights, method='weighted_boxes_fusion', iou_thr=0.5,
                    skip_box_thr=0.0001,
                    sigma=0.1):
    assert len(ipt_json_paths) == len(weights)
    img_info_dicts = mmcv.load(img_ann_path)['images']
    img_info_dict = dict()
    for img_info in img_info_dicts:
        img_info_dict[img_info['id']] = img_info

    res_dicts = []
    res_dict = dict()
    for json_path in ipt_json_paths:
        res_dicts.append(json_to_lisdict(json_path))
    for img_id in res_dicts[0]:
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(res_dicts)):
            boxes_list.append(res_dicts[i][img_id]['boxes'])
            scores_list.append(res_dicts[i][img_id]['scores'])
            labels_list.append(res_dicts[i][img_id]['labels'])

        if method == 'nms':
            boxes, scores, labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
        elif method == 'soft_nms':
            boxes, scores, labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr,
                                             sigma=sigma, thresh=skip_box_thr)
        elif method == 'non_maximum_weighted':
            boxes, scores, labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights,
                                                         iou_thr=iou_thr,
                                                         skip_box_thr=skip_box_thr)
        else:
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                          iou_thr=iou_thr,
                                                          skip_box_thr=skip_box_thr)
        res_dict[img_id] = dict(boxes=boxes, scores=scores, labels=labels)

    lisdict_to_json(res_dict, opt_json_path)


if __name__ == '__main__':
    ipt_paths = [
        '',
        '',
    ]
    opt_path = ''
    ensemble_models(ipt_paths, opt_path, weights=[1, 1])
