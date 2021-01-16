_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_icdar2021.py',
    '../_base_/datasets/icdar2021_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='DetectoRS_ResNet',
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True)))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
