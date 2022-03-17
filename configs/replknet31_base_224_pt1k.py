# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
from basecls.layers import NORM_TYPES

from basecls.configs.base_cfg import BaseConfig


_cfg = dict(
    batch_size=128,
    output_dir=None,
    model=dict(
        name="replknet31_base",
        drop_path_rate=0.5,
    ),
    bn=dict(
        precise_every_n_epoch=5,
    ),
    ## MegEngine use BGR as default colorspace
    # preprocess=dict(
    #     img_color_space="RGB",
    #     # flip from BGR mean & std to RGB
    #     img_mean=[103.530, 116.280, 123.675][::-1],
    #     img_std=[57.375, 57.12, 58.395][::-1],
    # ),
    test=dict(
       crop_pct=0.875,
    ),
    eval_every_n_epoch=5,
    loss=dict(
        label_smooth=0.1,
    ),
    augments=dict(
        name="RandAugment",
        rand_aug=dict(
            magnitude=9,
        ),
        resize=dict(
            interpolation="bicubic",
        ),
        rand_erase=dict(
            prob=0.25,
            mode="pixel",
        ),
        mixup=dict(
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
        ),
    ),
    data=dict(
        train_path="/path/to/imagenet/train",
        val_path="/path/to/imagenet/val",
        num_workers=10,
    ),
    solver=dict(
        optimizer="adamw",
        # `basic_lr` is the learning rate for a single GPU
        # 4e-3 per 2048 batch size == 2.5e-4 per 128 batch size
        basic_lr=2.5e-4,
        lr_min_factor=1e-3,
        weight_decay=(
            (0, "bias"),
            (0, NORM_TYPES),
            0.05,
        ),
        max_epoch=300,
        warmup_epochs=10,
        warmup_factor=0.1,
        lr_schedule="cosine",
    ),
    model_ema=dict(
        enabled=True,
        momentum=0.9992,
        update_period=8,
    ),
    fastrun=False,
    dtr=False,
    amp=dict(
        enabled=True,
        dynamic_scale=True,
    ),
    save_every_n_epoch=50,
)


class Cfg(BaseConfig):
    def __init__(self, values_or_file=None, **kwargs):
        super().__init__(_cfg)
        self.merge(values_or_file, **kwargs)
