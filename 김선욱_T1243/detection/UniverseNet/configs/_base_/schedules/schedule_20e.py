# optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 19])
# runner = dict(type='EpochBasedRunner', max_epochs=20)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.001,
    step=[10, 15, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)