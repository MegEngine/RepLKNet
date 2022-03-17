# RepLKNet (CVPR 2022)

This is the official MegEngine implementation of **RepLKNet**, from the following CVPR-2022 paper:

Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs.

The paper is released on arXiv: https://arxiv.org/abs/2203.06717.

## Other implementations

| framework | link |
|:---:|:---:|
|PyTorch (official)|https://github.com/DingXiaoH/RepLKNet-pytorch|
|Tensorflow| re-implementations are welcomed |
|PaddlePaddle  | re-implementations are welcomed |
| ... | |

## Catalog
- [x] Model code
- [x] MegEngine pretrained models
- [x] MegEngine training code
- [ ] MegEngine downstream models
- [ ] MegEngine downstream code

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models

### ImageNet-1K Models

| name | resolution |acc | #params | FLOPs | download |
|:---:|:---:|:---:|:---:| :---:|:---:|
| RepLKNet-31B | 224x224 | 83.58 | 79M | 15.3G | [0de394](https://data.megengine.org.cn/research/replknet/replknet31_base_224_pt1k_basecls.pkl) |


### ImageNet-22K Models

| name | resolution |acc | #params | FLOPs | 22K model | 1K model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|



### MegData-73M Models
| name | resolution |acc@1 | #params | FLOPs | MegData-73M model | 1K model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|


## Installation of MegEngine
```bash
pip3 install megengine -f https://megengine.org.cn/whl/mge.html --user
```
For more details, please check the [HomePage](https://github.com/MegEngine/MegEngine).

## Installation of BaseCls

[BaseCls](https://github.com/megvii-research/basecls) is an image classification framework built upon MegEngine.
We ultilize BaseCls for ImageNet pretraining and finetuning.

```bash
pip3 install basecls --user
```

Training and evaluation are configured through file. All default configurations are listed [here](https://github.com/megvii-research/basecls/blob/main/basecls/configs/base_cfg.py).

## Evaluation
```bash
./main_imagenet_test.py -f configs/config_replknet31_base.py -w [weights] batch_size 64 data.val_path /path/to/imagenet/val
```

## Training
```bash
./main_imagenet_train.py -f configs/config_replknet31_base.py data.train_path /path/to/imagenet/train data.val_path /path/to/imagenet/val
```

## Benchmark large depth-wise kernels

We can compare the kernel speed of MegEngine against PyTorch. A minimum version of megengine 1.8.2 is required for
optimized large depth-wise convolutions.

```bash
./main_benchmark.py
```

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
