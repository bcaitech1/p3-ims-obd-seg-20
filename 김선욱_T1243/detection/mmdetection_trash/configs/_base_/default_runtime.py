checkpoint_config = dict(max_keep_ckpts=1, interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        dict(

            type='WandbLoggerHook',

            init_kwargs=dict(#이 argument는 wandb.init()의 인자로 들어갈 값을 정의합니다

                project='프로젝트 이름 아무나',

                name = 'run 이름 아무나' #run name 

                # 그외 설정(자세한 사항은 wandb.init()의 parameter 참조

            ),

            with_step=False

        ),
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
