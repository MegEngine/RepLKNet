#!/usr/bin/env python3
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
"""
ImageNet testing script modifiled from BaseCls
https://github.com/megvii-research/basecls/blob/main/basecls/tools/cls_test.py
"""
import argparse
import importlib
import multiprocessing as mp
import os
import sys

import megengine as mge
import megengine.distributed as dist
from basecore.config import ConfigDict
from loguru import logger

from basecls.engine import ClsTester
from basecls.models import build_model, load_model
from basecls.utils import default_logging, registers, set_nccl_env, set_num_threads, setup_logger

from model_replknet import RepLKNet


def make_parser() -> argparse.ArgumentParser:
    """Build args parser for testing script.

    Returns:
        The args parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, help="testing process description file")
    parser.add_argument("-w", "--weight_file", default=None, type=str, help="weight file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="other options")
    return parser


@logger.catch
def worker(args: argparse.Namespace):
    """Worker function for testing script.

    Args:
        args: args for testing script.
    """
    logger.info(f"Init process group for gpu{dist.get_rank()} done")

    sys.path.append(os.path.dirname(args.file))
    module_name = os.path.splitext(os.path.basename(args.file))[0]
    current_network = importlib.import_module(module_name)
    cfg = current_network.Cfg()
    if cfg.output_dir is None:
        cfg.output_dir = f"./logs_{module_name}"
    cfg.output_dir = os.path.abspath(cfg.output_dir)

    if args.weight_file:
        cfg.weights = args.weight_file
    else:
        cfg.weights = os.path.join(cfg.output_dir, "latest.pkl")

    cfg.merge(args.opts)
    cfg.set_mode("freeze")

    if dist.get_rank() == 0 and not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    dist.group_barrier()

    setup_logger(cfg.output_dir, "test_log.txt", to_loguru=True)
    logger.info(f"args: {args}")

    if cfg.fastrun:
        logger.info("Using fastrun mode...")
        mge.functional.debug_param.set_execution_strategy("PROFILE")

    tester = build(cfg)
    tester.test()


def build(cfg: ConfigDict):
    """Build function for testing script.

    Args:
        cfg: config for testing.

    Returns:
        A tester.
    """
    model = build_model(cfg)
    load_model(model, cfg.weights)

    if isinstance(model, RepLKNet):
        model = RepLKNet.convert_to_deploy(model)

    default_logging(cfg, model)

    dataloader = registers.dataloaders.get(cfg.data.name).build(cfg, False)
    # FIXME: need atomic user_pop, maybe in MegEngine 1.5?
    # tester = BaseTester(model, dataloader, AccEvaluator())
    return ClsTester(cfg, model, dataloader)


def main():
    """Main function for testing script."""
    parser = make_parser()
    args = parser.parse_args()

    mp.set_start_method("spawn")

    set_nccl_env()
    set_num_threads()

    if not os.path.exists(args.file):
        raise ValueError("Description file does not exist")

    device_count = mge.device.get_device_count("gpu")

    if device_count == 0:
        logger.warning("No GPU was found, testing on CPU")
        worker(args)
    elif device_count > 1:
        mp_worker = dist.launcher(worker)
        mp_worker(args)
    else:
        worker(args)


if __name__ == "__main__":
    main()
