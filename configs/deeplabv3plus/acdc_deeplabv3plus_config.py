_base_ = [
    '../_base_/models/my_deeplabv3plus.py',
    '../_base_/datasets/cityscapes_all.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 1024)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
