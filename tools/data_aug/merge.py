from pycocotools.coco import COCO
import json
path = "/home/zhouyuxuan/ICDAR2021/"

CLASSES = ('embedded', 'isolated')

gen_coco = COCO(path + "gen.json")
gen_img_ids = gen_coco.get_img_ids()
gen_images = gen_coco.load_imgs(gen_img_ids)
gen_ann_ids = gen_coco.get_ann_ids(img_ids=gen_img_ids)
gen_annotations = gen_coco.load_anns(gen_ann_ids)
max_img_id=max(gen_img_ids)+1
max_ann_id=max(gen_ann_ids)+1
# for i in range(len(gen_images)):
#     gen_images[i]['file_name']="gen/"+gen_images[i]['file_name']

ori_coco=COCO(path+"TrM.json")
ori_img_ids = ori_coco.get_img_ids()
ori_images = ori_coco.load_imgs(ori_img_ids)
ori_ann_ids = ori_coco.get_ann_ids(img_ids=ori_img_ids)
ori_annotations = ori_coco.load_anns(ori_ann_ids)
for i in range(len(ori_images)):
    # ori_images[i]['file_name']="TrM/"+ori_images[i]['file_name']
    ori_images[i]['id']+=max_img_id
for i in range(len(ori_annotations)):
    ori_annotations[i]['image_id']+=max_img_id
    ori_annotations[i]['id']+=max_ann_id

gen_images.extend(ori_images)
gen_annotations.extend(ori_annotations)
json_file = open(path + "merged.json", "w")
json_file.write(json.dumps({'images': gen_images,
                            'annotations': gen_annotations,
                            'categories': [{'id': 0, 'name': 'embedded'}, {'id': 1, 'name': 'isolated'}]}))
json_file.close()
