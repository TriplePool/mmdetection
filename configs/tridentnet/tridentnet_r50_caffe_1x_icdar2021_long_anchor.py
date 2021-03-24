_base_ = [
    '../_base_/models/faster_rcnn_r50_caffe_c4_icdar2021.py',
    '../_base_/datasets/icdar2021_detection.py',
    '../_base_/schedules/schedule_1x_half_batch.py', '../_base_/default_runtime.py'
]

model = dict(
    type='TridentFasterRCNN',
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    backbone=dict(
        type='TridentResNet',
        trident_dilations=(1, 2, 3),
        num_branch=3,
        test_branch_idx=1),
    rpn_head=dict(
        type='RPNHead',
        in_channels=1024,
        feat_channels=1024,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[2, 4, 8, 16, 32],
            ratios=[0.5, 1.0, 2.0, 4.0],
            strides=[16]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(type='TridentRoIHead', num_branch=3, test_branch_idx=1))

train_cfg = dict(
    rpn_proposal=dict(nms_post=500, max_num=500),
    rcnn=dict(
        sampler=dict(num=128, pos_fraction=0.5, add_gt_as_proposals=False)))

# use caffe img_norm
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1447, 2048), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1447, 2048),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
