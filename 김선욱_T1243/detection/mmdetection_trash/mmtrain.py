from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)


classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# config file 들고오기
# cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
cfg = Config.fromfile('./configs/cascade_rcnn/cascade_rcnn_x101_32x4d_fpn_1x_coco.py')

PREFIX = '../../input/data/'


# dataset 바꾸기
cfg.data.train.classes = classes
cfg.data.train.img_prefix = PREFIX
cfg.data.train.ann_file = PREFIX + 'train.json'
cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

print(cfg.data.train)
cfg.data.val.classes = classes
cfg.data.val.img_prefix = PREFIX
cfg.data.val.ann_file = PREFIX + 'val.json'
cfg.data.val.pipeline[1]['img_scale'] = (512, 512)


cfg.data.test.classes = classes
cfg.data.test.img_prefix = PREFIX
cfg.data.test.ann_file = PREFIX + 'test.json'
cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

cfg.data.samples_per_gpu = 4

cfg.seed=2020
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/cascade_rcnn_101'

# cfg.model.roi_head.bbox_head.num_classes = 11

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")
cfg.log_config.hooks[1].init_kwargs.project = 'cascade_rcnn'
cfg.log_config.hooks[1].init_kwargs.name = 'c_0_beta'

print(cfg.model)




model = build_detector(cfg.model)
datasets = [build_dataset(cfg.data.train)]

train_detector(model, datasets[0], cfg, distributed=False, validate=True)


