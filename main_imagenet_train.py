#!/usr/bin/env python3
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
"""
ImageNet training script modifiled from BaseCls
https://github.com/megvii-research/basecls/blob/main/basecls/tools/cls_train.py
"""
import argparse
import importlib
import os
import sys

import megengine as mge
import megengine.distributed as dist
from basecore.config import ConfigDict
from loguru import logger

from basecls.models import build_model, load_model, sync_model
from basecls.utils import registers, set_nccl_env, set_num_threads, setup_logger

import model_replknet


def default_parser() -> argparse.ArgumentParser:
    """Build args parser for training script.

    Returns:
        The args parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="training process description file")
    parser.add_argument(
        "--resume", action="store_true", help="resume training from saved checkpoint or not"
    )
    parser.add_argument(
        "opts",
        default=None,
        help="Modify config options using the command-line",
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def worker(args: argparse.Namespace):
    """Worker function for training script.

    Args:
        args: args for training script.
    """
    logger.info(f"Init process group for gpu{dist.get_rank()} done")

    sys.path.append(os.path.dirname(args.file))
    module_name = os.path.splitext(os.path.basename(args.file))[0]
    current_network = importlib.import_module(module_name)
    cfg = current_network.Cfg()
    cfg.merge(args.opts)
    cfg.resume = args.resume
    if cfg.output_dir is None:
        cfg.output_dir = f"./logs_{module_name}"
    cfg.output_dir = os.path.abspath(cfg.output_dir)

    cfg.set_mode("freeze")

    if dist.get_rank() == 0 and not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    dist.group_barrier()

    setup_logger(cfg.output_dir, "train_log.txt", to_loguru=True)
    logger.info(f"args: {args}")

    if cfg.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    if cfg.dtr:
        logger.info("Enabling DTR...")
        mge.dtr.enable()

    trainer = build(cfg)
    trainer.train()


def build(cfg: ConfigDict):
    """Build function for training script.

    Args:
        cfg: config for training.

    Returns:
        A trainer.
    """
    model = build_model(cfg)
    if getattr(cfg, "weights", None) is not None:
        load_model(model, cfg.weights, strict=False)
    sync_model(model)
    model.train()

    logger.info(f"Using augments named {cfg.augments.name}")
    augments = registers.augments.get(cfg.augments.name).build(cfg)
    logger.info(f"Using dataloader named {cfg.data.name}")
    dataloader = registers.dataloaders.get(cfg.data.name).build(cfg, True, augments)
    logger.info(f"Using solver named {cfg.solver.name}")
    solver = registers.solvers.get(cfg.solver.name).build(cfg, model)
    logger.info(f"Using hooks named {cfg.hooks_name}")
    hooks = registers.hooks.get(cfg.hooks_name).build(cfg)

    logger.info(f"Using trainer named {cfg.trainer_name}")
    TrainerClass = registers.trainers.get(cfg.trainer_name)
    return TrainerClass(cfg, model, dataloader, solver, hooks=hooks)


def main():
    """Main function for training script."""
    parser = default_parser()
    args = parser.parse_args()

    set_nccl_env()
    set_num_threads()

    device_count = mge.device.get_device_count("gpu")
    launcher = dist.launcher

    if not os.path.exists(args.file):
        raise ValueError("Description file does not exist")

    if device_count == 0:
        raise ValueError("Number of devices should be greater than 0")
    elif device_count > 1 or os.environ.get("RLAUNCH_REPLICA_TOTAL", 0) > 1:
        mp_worker = launcher(worker)
        mp_worker(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
