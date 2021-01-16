_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn_icdar2021.py',
    '../_base_/datasets/icdar2021_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2)

# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=10, norm_type=2))
