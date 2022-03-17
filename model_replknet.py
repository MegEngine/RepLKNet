# Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import copy

import megengine
import megengine.functional as F
import megengine.module as nn
import numpy as np
from basecls.layers import DropPath, init_weights
from basecls.utils import registers


def _fuse_prebn_conv1x1(bn, conv):
    module_output = copy.deepcopy(conv)
    module_output.bias = megengine.Parameter(np.zeros(module_output._infer_bias_shape(), dtype=np.float32))
    assert conv.groups == 1
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = F.sqrt(running_var + eps)
    t = (gamma / std).reshape(1, -1, 1, 1)
    module_output.weight[:] = kernel * t
    module_output.bias[:] = F.conv2d(beta - running_mean * gamma / std, kernel, conv.bias)
    return module_output


def _fuse_conv_bn(conv, bn):
    module_output = copy.deepcopy(conv)
    module_output.bias = megengine.Parameter(np.zeros(module_output._infer_bias_shape(), dtype=np.float32))
    # flatten then reshape in case of group conv
    kernel = F.flatten(conv.weight, end_axis=conv.weight.ndim - 4)
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = F.sqrt(running_var + eps)
    t = (gamma / std).reshape(-1, 1, 1, 1)
    module_output.weight[:] = (kernel * t).reshape(module_output.weight.shape)
    module_output.bias[:] = beta + ((conv.bias if conv.bias is not None else 0) - running_mean) * gamma / std
    return module_output


class ConvBn2d(nn.ConvBn2d):
    def __init__(self, *args, **kwargs):
        bias = kwargs.pop("bias", False) and False
        super().__init__(*args, bias=bias, **kwargs)

    @classmethod
    def fuse_conv_bn(cls, module: nn.Module):
        module_output = module
        if isinstance(module, ConvBn2d):
            return _fuse_conv_bn(module.conv, module.bn)
        for name, child in module.named_children():
            setattr(module_output, name, cls.fuse_conv_bn(child))
        del module
        return module_output


class LargeKernelReparam(nn.Module):
    def __init__(self, channels, kernel, small_kernels=()):
        super(LargeKernelReparam, self).__init__()
        self.dw_large = ConvBn2d(channels, channels, kernel, padding=kernel // 2, groups=channels)

        self.small_kernels = small_kernels
        for k in self.small_kernels:
            setattr(self, f"dw_small_{k}", ConvBn2d(channels, channels, k, padding=k // 2, groups=channels))

    def forward(self, inp):
        outp = self.dw_large(inp)
        for k in self.small_kernels:
            outp += getattr(self, f"dw_small_{k}")(inp)
        return outp

    @classmethod
    def convert_to_deploy(cls, module: nn.Module):
        module_output = module
        if isinstance(module, LargeKernelReparam):
            module = ConvBn2d.fuse_conv_bn(module)
            module_output = copy.deepcopy(module.dw_large)
            kernel = module_output.kernel_size[0]
            for k in module.small_kernels:
                dw_small = getattr(module, f"dw_small_{k}")
                module_output.weight += F.pad(dw_small.weight, [[0, 0]] * 3 + [[(kernel - k) // 2] * 2] * 2)
                module_output.bias += dw_small.bias
            return module_output
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


class Mlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.,):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.fc1 = ConvBn2d(in_channels, hidden_features, 1, stride=1, padding=0)
        self.act = act_layer()
        self.fc2 = ConvBn2d(hidden_features, out_features, 1, stride=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepLKBlock(nn.Module):

    def __init__(self, channels, kernel, small_kernels=(), dw_ratio=1.0, mlp_ratio=4.0, drop_path=0., activation=nn.ReLU):
        super().__init__()

        self.pre_bn = nn.BatchNorm2d(channels)
        self.pw1 = ConvBn2d(channels, int(channels * dw_ratio), 1, 1, 0)
        self.pw1_act = activation()
        self.dw = LargeKernelReparam(int(channels * dw_ratio), kernel, small_kernels=small_kernels)
        self.dw_act = activation()
        self.pw2 = ConvBn2d(int(channels * dw_ratio), channels, 1, 1, 0)

        self.premlp_bn = nn.BatchNorm2d(channels)
        self.mlp = Mlp(in_channels=channels, hidden_channels=int(channels * mlp_ratio))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        y = self.pre_bn(x)
        y = self.pw1_act(self.pw1(y))
        y = self.dw_act(self.dw(y))
        y = self.pw2(y)
        x = x + self.drop_path(y)

        y = self.premlp_bn(x)
        y = self.mlp(y)
        x = x + self.drop_path(y)

        return x

    @classmethod
    def convert_to_deploy(cls, module: nn.Module):
        module_output = module
        if isinstance(module, RepLKBlock):
            LargeKernelReparam.convert_to_deploy(module)
            ConvBn2d.fuse_conv_bn(module)

            module.pre_bn, module.pw1 = nn.Identity(), _fuse_prebn_conv1x1(module.pre_bn, module.pw1)
            module.premlp_bn, module.mlp.fc1 = nn.Identity(), _fuse_prebn_conv1x1(module.premlp_bn, module.mlp.fc1)
            return module_output
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


class DownSample(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__(
            ConvBn2d(in_channels, out_channels, 1),
            activation(),
            ConvBn2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels),
            activation(),
        )


class Stem(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation=nn.ReLU):
        super().__init__(
            ConvBn2d(in_channels, out_channels, 3, stride=2, padding=1),
            activation(),
            ConvBn2d(out_channels, out_channels, 3, padding=1, groups=out_channels),
            activation(),
            ConvBn2d(out_channels, out_channels, 1),
            activation(),
            ConvBn2d(out_channels, out_channels, 3, stride=2, padding=1, groups=out_channels),
            activation(),
        )

class RepLKNet(nn.Module):

    def __init__(
        self,
        in_channels=3,
        depths=(2, 2, 18, 2),
        dims=(128, 256, 512, 1024),
        kernel_sizes=(31, 29, 27, 13),
        small_kernels=(5,),
        dw_ratio=1.0,
        mlp_ratio=4.0,
        num_classes=1000,
        drop_path_rate=0.5,
    ):
        super().__init__()

        self.stem = Stem(in_channels, dims[0])
        # stochastic depth
        dpr = (x for x in np.linspace(0, drop_path_rate, sum(depths)))  # stochastic depth decay rule

        self.blocks = []

        for stage, (depth, dim, ksize) in enumerate(zip(depths, dims, kernel_sizes)):
            for _ in range(depth):
                self.blocks.append(
                    RepLKBlock(dim, ksize, small_kernels=small_kernels,
                        dw_ratio=dw_ratio, mlp_ratio=mlp_ratio, drop_path=next(dpr))
                )
            if stage < len(depths) - 1:
                self.blocks.append(DownSample(dim, dims[stage + 1]))

        self.norm = nn.BatchNorm2d(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        init_weights(self)

    def forward_features(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = F.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    @classmethod
    def convert_to_deploy(cls, module: nn.Module):
        module_output = module
        if isinstance(module, RepLKNet):
            RepLKBlock.convert_to_deploy(module)
            ConvBn2d.fuse_conv_bn(module)
            return module_output
        for name, child in module.named_children():
            setattr(module_output, name, cls.convert_to_deploy(child))
        del module
        return module_output


@registers.models.register()
def replknet31_base(**kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(128, 256, 512, 1024), dw_ratio=1.0, **kwargs)


@registers.models.register()
def replknet31_large(**kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(192, 384, 768, 1536), dw_ratio=1.0, **kwargs)


@registers.models.register()
def replknet_xlarge(**kwargs):
    kwargs.pop("head", None)
    return RepLKNet(dims=(256, 512, 1024, 2048), kernel_sizes=(27, 27, 27, 13), small_kernels=(), dw_ratio=1.5, **kwargs)
