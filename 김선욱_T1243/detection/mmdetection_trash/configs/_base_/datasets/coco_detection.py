dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# A.OneOf( [A.RandomRotate90(),
#                 A.HorizontalFlip()],
#                 p=1.0),
#         A.ShiftScaleRotate(p=0.25),

albu_train_transforms = [
    
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='Rotate',
                limit=10,
                p=0.6),
            dict(
                type='HorizontalFlip',
                p=0.6)
        ],
        p=0.1),
    
    dict(
        type='ShiftScaleRotate',
        p=0.5),
]



# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(512,512), keep_ratio=True),
#     dict(type='Pad', size_divisor=32),
#     dict(type='RandomFlip', flip_ratio = 0.5),
    
#     # dict(
#     #     type='Albu',
#     #     transforms=albu_train_transforms,
#     #     bbox_params=dict(
#     #         type='BboxParams',
#     #         format='pascal_voc',
#     #         label_fields=['gt_labels'],
#     #         min_visibility=0.0,
#     #         filter_lost_elements=True),
#     #     keymap={
#     #         'img': 'image',
#     #         'gt_bboxes': 'bboxes'
#     #     },
#     #     update_pad_shape=False,
#     #     skip_img_without_anno=True),
#     # dict(type='Normalize', **img_norm_cfg),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
train_pipeline = [

    dict(type='LoadImageFromFile'),

    dict(type='LoadAnnotations', with_bbox=True),

    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),

    dict(type='Pad', size_divisor=32),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),

    dict(type='Normalize', **img_norm_cfg),

    

    dict(type='DefaultFormatBundle'),

    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),

]
# train_pipeline = [

#     dict(type='LoadImageFromFile'),

#     dict(type='LoadAnnotations', with_bbox=True),

#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),

#     dict(type='RandomFlip', flip_ratio=0.5),

#     dict(type='Normalize', **img_norm_cfg),

#     dict(type='Pad', size_divisor=32),

#     dict(type='DefaultFormatBundle'),

#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),

# ]

test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(512, 512),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Normalize', **img_norm_cfg),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img']),
                ])
        ]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric="bbox", save_best="bbox_mAP_50")


