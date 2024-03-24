# dataset settings
dataset_type = 'CityscapesDataset'
data_root = 'H:/ACDC/cityscapes_acdc_all/'
crop_size = (512, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
# train dataset
train_fog=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train/fog', seg_map_path='gtFine/train/fog'),
        pipeline=train_pipeline)
train_night=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train/night', seg_map_path='gtFine/train/night'),
        pipeline=train_pipeline)
train_rain=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train/rain', seg_map_path='gtFine/train/rain'),
        pipeline=train_pipeline)
train_snow=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/train/snow', seg_map_path='gtFine/train/snow'),
        pipeline=train_pipeline)
# val dataset
val_fog=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val/fog', seg_map_path='gtFine/val/fog'),
        pipeline=test_pipeline)
val_night=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val/night', seg_map_path='gtFine/val/night'),
        pipeline=test_pipeline)
val_rain=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val/rain', seg_map_path='gtFine/val/rain'),
        pipeline=test_pipeline)
val_snow=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/val/snow', seg_map_path='gtFine/val/snow'),
        pipeline=test_pipeline)

# test dataset
test_fog=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/test/fog', seg_map_path='gtFine/test/fog'),
        pipeline=test_pipeline)
test_night=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/test/night', seg_map_path='gtFine/test/night'),
        pipeline=test_pipeline)
test_rain=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/test/rain', seg_map_path='gtFine/test/rain'),
        pipeline=test_pipeline)
test_snow=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='leftImg8bit/test/snow', seg_map_path='gtFine/test/snow'),
        pipeline=test_pipeline)
# dataloader
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(type='ConcatDataset',datasets=[train_night,train_fog,train_snow,train_rain]) )

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset',datasets=[val_night,val_fog,val_snow,val_rain]))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='ConcatDataset',datasets=[val_night])  # modify
)
# val_fog,val_snow,val_rain
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
