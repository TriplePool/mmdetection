_base_ = './mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_icdar2021_embedded.py'
# learning policy
lr_config = dict(step=[16, 23])
total_epochs = 24
