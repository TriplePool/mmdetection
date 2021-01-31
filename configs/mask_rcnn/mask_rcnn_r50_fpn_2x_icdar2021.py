_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_icdar2021.py',
    '../_base_/datasets/icdar2021_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
