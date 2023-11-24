# RemoteNet
Paper:[RemoteNet: Remote Sensing Image Segmentation Network based on Global-Local Information](https://arxiv.org/abs/2302.13084)

Because the original paper does not have open source code, I tried to reproduce it.
## This project reproduces this paper

# Model summary
```python
===============================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
===============================================================================================
RemoteNet                                     [1, 6, 256, 256]          --
├─Encoder: 1-1                                [1, 32, 64, 64]           --
│    └─stage: 2-1                             [1, 32, 64, 64]           --
│    │    └─OverlapPatchEmbed: 3-1            [1, 4096, 32]             4,800
│    │    └─Sequential: 3-2                   [1, 4096, 32]             37,824
│    └─stage: 2-2                             [1, 64, 32, 32]           --
│    │    └─OverlapPatchEmbed: 3-3            [1, 1024, 64]             100,544
│    │    └─Sequential: 3-4                   [1, 1024, 64]             199,168
│    └─stage: 2-3                             [1, 160, 16, 16]          --
│    │    └─OverlapPatchEmbed: 3-5            [1, 256, 160]             502,240
│    │    └─Sequential: 3-6                   [1, 256, 160]             1,852,800
│    └─stage: 2-4                             [1, 256, 8, 8]            --
│    │    └─OverlapPatchEmbed: 3-7            [1, 64, 256]              2,007,808
│    │    └─Sequential: 3-8                   [1, 64, 256]              2,366,976
├─Decoder: 1-2                                [1, 6, 256, 256]          --
│    └─Conv2d: 2-5                            [1, 64, 8, 8]             16,448
│    └─Block: 2-6                             [1, 64, 8, 8]             --
│    │    └─BatchNorm2d: 3-9                  [1, 64, 8, 8]             128
│    │    └─GlobalLocalAttention: 3-10        [1, 64, 8, 8]             63,816
│    │    └─Identity: 3-11                    [1, 64, 8, 8]             --
│    │    └─BatchNorm2d: 3-12                 [1, 64, 8, 8]             128
│    │    └─Mlp: 3-13                         [1, 64, 8, 8]             33,088
│    │    └─Identity: 3-14                    [1, 64, 8, 8]             --
│    │    └─ConvBN: 3-15                      [1, 64, 8, 8]             4,224
│    │    └─ConvBN: 3-16                      [1, 64, 8, 8]             36,992
│    │    └─ConvBN: 3-17                      [1, 64, 8, 8]             102,528
│    │    └─SeparableConvBN: 3-18             [1, 64, 8, 8]             7,360
│    └─Fusion: 2-7                            [1, 64, 16, 16]           --
│    │    └─Conv2d: 3-19                      [1, 64, 16, 16]           10,304
│    │    └─Sequential: 3-20                  [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-21                  [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-22                  [1, 64, 16, 16]           36,992
│    │    └─ConvBNReLU: 3-23                  [1, 64, 16, 16]           (recursive)
│    └─Block: 2-8                             [1, 64, 16, 16]           (recursive)
│    │    └─BatchNorm2d: 3-24                 [1, 64, 16, 16]           (recursive)
│    │    └─GlobalLocalAttention: 3-25        [1, 64, 16, 16]           (recursive)
│    │    └─Identity: 3-26                    [1, 64, 16, 16]           --
│    │    └─BatchNorm2d: 3-27                 [1, 64, 16, 16]           (recursive)
│    │    └─Mlp: 3-28                         [1, 64, 16, 16]           (recursive)
│    │    └─Identity: 3-29                    [1, 64, 16, 16]           --
│    │    └─ConvBN: 3-30                      [1, 64, 16, 16]           (recursive)
│    │    └─ConvBN: 3-31                      [1, 64, 16, 16]           (recursive)
│    │    └─ConvBN: 3-32                      [1, 64, 16, 16]           (recursive)
│    │    └─SeparableConvBN: 3-33             [1, 64, 16, 16]           (recursive)
│    └─Fusion: 2-9                            [1, 64, 32, 32]           --
│    │    └─Conv2d: 3-34                      [1, 64, 32, 32]           4,160
│    │    └─Sequential: 3-35                  [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-36                  [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-37                  [1, 64, 32, 32]           36,992
│    │    └─ConvBNReLU: 3-38                  [1, 64, 32, 32]           (recursive)
│    └─Block: 2-10                            [1, 64, 32, 32]           (recursive)
│    │    └─BatchNorm2d: 3-39                 [1, 64, 32, 32]           (recursive)
│    │    └─GlobalLocalAttention: 3-40        [1, 64, 32, 32]           (recursive)
│    │    └─Identity: 3-41                    [1, 64, 32, 32]           --
│    │    └─BatchNorm2d: 3-42                 [1, 64, 32, 32]           (recursive)
│    │    └─Mlp: 3-43                         [1, 64, 32, 32]           (recursive)
│    │    └─Identity: 3-44                    [1, 64, 32, 32]           --
│    │    └─ConvBN: 3-45                      [1, 64, 32, 32]           (recursive)
│    │    └─ConvBN: 3-46                      [1, 64, 32, 32]           (recursive)
│    │    └─ConvBN: 3-47                      [1, 64, 32, 32]           (recursive)
│    │    └─SeparableConvBN: 3-48             [1, 64, 32, 32]           (recursive)
│    └─Fusion: 2-11                           [1, 64, 64, 64]           --
│    │    └─Conv2d: 3-49                      [1, 64, 64, 64]           2,112
│    │    └─Sequential: 3-50                  [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-51                  [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-52                  [1, 64, 64, 64]           36,992
│    │    └─ConvBNReLU: 3-53                  [1, 64, 64, 64]           (recursive)
│    └─FRM: 2-12                              [1, 6, 256, 256]          --
│    │    └─SeparableConvBN: 3-54             [1, 64, 64, 64]           4,800
│    │    └─SeparableConvBN: 3-55             [1, 64, 64, 64]           5,824
│    │    └─ConvBN: 3-56                      [1, 64, 64, 64]           4,224
│    │    └─ConvBNReLU: 3-57                  [1, 64, 64, 64]           36,992
│    │    └─SeparableConvBNReLU: 3-58         [1, 64, 64, 64]           103,104
│    │    └─ConvBN: 3-59                      [1, 64, 64, 64]           (recursive)
│    │    └─ConvBNReLU: 3-60                  [1, 64, 64, 64]           4,224
│    │    └─ConvBNReLU: 3-61                  [1, 64, 64, 64]           (recursive)
│    │    └─Conv2d: 3-62                      [1, 6, 64, 64]            390
===============================================================================================
Total params: 7,748,014
Trainable params: 7,748,014
Non-trainable params: 0
Total mult-adds (G): 2.03
===============================================================================================
Input size (MB): 0.79
Forward/backward pass size (MB): 132.40
Params size (MB): 30.98
Estimated Total Size (MB): 164.17
===============================================================================================
```

# 参考
[https://github.com/WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg)
