_base_ = './vfnet_r50_fpn_mstrain_2x_icdar2021.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    bbox_head=dict(dcn_on_last_conv=True))

dataset_type = 'CocoDataset'
# data_root = '/home/weibaole/disk1/gpu/Workspace/Datas/ICDAR2021/'
data_root = '/home/wbl/workspace/data/ICDAR2021/'

classes = ('isolated',)
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'TrM_isolated.json',
        img_prefix=data_root + 'TrM/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VaM_isolated.json',
        img_prefix=data_root + 'VaM/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VaM_isolated.json',
        img_prefix=data_root + 'VaM/',
        classes=classes))
