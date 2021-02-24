_base_ = './vfnet_r50_fpn_mstrain_2x_icdar2021.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(dcn_on_last_conv=True))

dataset_type = 'Icdar2021Dataset'
# data_root = '/home/weibaole/disk1/gpu/Workspace/Datas/ICDAR2021/'
data_root = '/home/wbl/workspace/data/ICDAR2021/'

classes = ('isolated',)
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'TrM_isolated.json',
        img_prefix=data_root + 'TrM_merge/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VaM_isolated.json',
        img_prefix=data_root + 'VaM_merge/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VaM_isolated.json',
        img_prefix=data_root + 'VaM_merge/',
        classes=classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(2000, 900), (2000, 600)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 724),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]
