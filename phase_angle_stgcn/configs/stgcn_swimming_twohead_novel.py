model = dict(
    type='RecognizerGCNTwoHead',
    backbone=dict(
        type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    error_head=dict(type='GCNHead', num_classes=4, in_channels=260),
    phase_head=dict(type='GCNHead', num_classes=4, in_channels=256),
    phase_loss_weight=0.3)
dataset_type = 'PoseDataset'
ann_file = 'data/swimming/datasets/swimming_4class_phase_with_r.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=24),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(
        type='Collect',
        keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=24, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(
        type='Collect',
        keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=24, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(
        type='Collect',
        keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
        meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='PoseDataset',
        ann_file='data/swimming/datasets/swimming_4class_phase_with_r.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=24),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(
                type='Collect',
                keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
        ],
        split='train'),
    val=dict(
        type='PoseDataset',
        ann_file='data/swimming/datasets/swimming_4class_phase_with_r.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=24, num_clips=1),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(
                type='Collect',
                keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
        ],
        split='val'),
    test=dict(
        type='PoseDataset',
        ann_file='data/swimming/datasets/swimming_4class_phase_with_r.pkl',
        pipeline=[
            dict(type='PreNormalize2D'),
            dict(type='GenSkeFeat', dataset='coco', feats=['j']),
            dict(type='UniformSample', clip_len=24, num_clips=10),
            dict(type='PoseDecode'),
            dict(type='FormatGCNInput', num_person=1),
            dict(
                type='Collect',
                keys=['keypoint', 'label', 'phase_label', 'cycle_r'],
                meta_keys=[]),
            dict(type='ToTensor', keys=['keypoint', 'label', 'phase_label', 'cycle_r'])
        ],
        split='test'))
optimizer = dict(
    type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=['mean_class_accuracy'],
    save_best='mean_class_accuracy',
    rule='greater')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcn/stgcn_swimming_twohead_angle_v2/j'
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)
