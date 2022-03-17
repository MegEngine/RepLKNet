#!/usr/bin/env python3
# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
"""Benchmark depth-wise large convolution for megengine and pytorch"""
import os
import time

import megengine
import megengine.functional as F
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

assert megengine.__version__ >= "1.8.2", "please update megengine via " \
"`pip3 megengine -f https://megengine.org.cn/whl/mge.html -U --user`"


def benchmark_megengine(batch_size, resolution, channels, depth, kernel_size, niters=10):
    input = F.ones([batch_size, channels, resolution, resolution]) * 1e-3
    weight = F.ones([channels, 1, 1, kernel_size, kernel_size]) * 1e-3
    diff = []
    for i in range(niters):
        x = input
        megengine._full_sync()
        t = time.perf_counter()
        for _ in range(depth):
            x = F.conv2d(x, weight, bias=None, padding=kernel_size // 2, groups=channels)
        megengine._full_sync()
        diff.append((time.perf_counter() - t) * 1000)
    diff = sum(sorted(diff)[1:-2]) / (niters - 3)
    print(f"benchmark_megeg\tB{batch_size},R{resolution},C{channels},D{depth},K{kernel_size}\t{diff:.3f} ms")
    return diff


@torch.no_grad()
def benchmark_torch(batch_size, resolution, channels, depth, kernel_size, niters=10):
    input = torch.randn(batch_size, channels, resolution, resolution).cuda() * 1e-3
    weight = torch.randn(channels, 1, kernel_size, kernel_size).cuda() * 1e-3
    diff = []
    for i in range(niters):
        x = input.clone()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(depth):
            x = torch.nn.functional.conv2d(x, weight, bias=None, padding=kernel_size // 2, groups=channels)
        torch.cuda.synchronize()
        diff.append((time.perf_counter() - t) * 1000)
    diff = sum(sorted(diff)[1:-2]) / (niters - 3)
    print(f"benchmark_torch\tB{batch_size},R{resolution},C{channels},D{depth},K{kernel_size}\t{diff:.3f} ms")
    return diff


if __name__ == "__main__":
    for resolution in (16, 32):
        for kernel_size in (7, 17, 27):
            benchmark_torch(64, resolution, 256, 12, kernel_size)
            benchmark_megengine(64, resolution, 256, 12, kernel_size)
