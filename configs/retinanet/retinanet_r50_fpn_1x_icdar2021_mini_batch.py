_base_ = [
    '../_base_/models/retinanet_r50_fpn_icdar2021.py',
    '../_base_/datasets/icdar2021_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)
