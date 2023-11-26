# RemoteNet
Paper:[RemoteNet: Remote Sensing Image Segmentation Network based on Global-Local Information](https://arxiv.org/abs/2302.13084)

Because the original paper does not have open source code, I tried to reproduce it.
## This project reproduces this paper
# Major changes
😒😒😒 Although the paper does not say it, the Encoder is [pvt_v2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/pvt_v2.py)
# Model summary
```python
====================================================================================================
Layer (type:depth-idx)                             Output Shape              Param #
====================================================================================================
RemoteNet                                          [1, 6, 768, 768]          --
├─Encoder: 1-1                                     [1, 32, 192, 192]         --
│    └─stage: 2-1                                  [1, 32, 192, 192]         --
│    │    └─OverlapPatchEmbed: 3-1                 [1, 36864, 32]            4,800
│    │    └─Sequential: 3-2                        [1, 36864, 32]            238,560
│    └─stage: 2-2                                  [1, 64, 96, 96]           --
│    │    └─OverlapPatchEmbed: 3-3                 [1, 9216, 64]             100,544
│    │    └─Sequential: 3-4                        [1, 9216, 64]             472,320
│    └─stage: 2-3                                  [1, 160, 48, 48]          --
│    │    └─OverlapPatchEmbed: 3-5                 [1, 2304, 160]            502,240
│    │    └─Sequential: 3-6                        [1, 2304, 160]            2,508,480
│    └─stage: 2-4                                  [1, 256, 24, 24]          --
│    │    └─OverlapPatchEmbed: 3-7                 [1, 576, 256]             2,007,808
│    │    └─Sequential: 3-8                        [1, 576, 256]             2,397,696
├─Decoder: 1-2                                     [1, 6, 768, 768]          --
│    └─Conv2d: 2-5                                 [1, 64, 24, 24]           16,448
│    └─Block: 2-6                                  [1, 64, 24, 24]           --
│    │    └─BatchNorm2d: 3-9                       [1, 64, 24, 24]           128
│    │    └─GlobalLocalAttention: 3-10             [1, 64, 24, 24]           63,816
│    │    └─Identity: 3-11                         [1, 64, 24, 24]           --
│    │    └─BatchNorm2d: 3-12                      [1, 64, 24, 24]           128
│    │    └─Mlp: 3-13                              [1, 64, 24, 24]           33,088
│    │    └─Identity: 3-14                         [1, 64, 24, 24]           --
│    │    └─ConvBN: 3-15                           [1, 64, 24, 24]           4,224
│    │    └─ConvBN: 3-16                           [1, 64, 24, 24]           36,992
│    │    └─ConvBN: 3-17                           [1, 64, 24, 24]           102,528
│    │    └─SeparableConvBN: 3-18                  [1, 64, 24, 24]           7,360
│    └─Fusion: 2-7                                 [1, 64, 48, 48]           --
│    │    └─Conv2d: 3-19                           [1, 64, 48, 48]           10,304
│    │    └─Sequential: 3-20                       [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-21                       [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-22                       [1, 64, 48, 48]           36,992
│    │    └─ConvBNReLU: 3-23                       [1, 64, 48, 48]           (recursive)
│    └─Block: 2-8                                  [1, 64, 48, 48]           (recursive)
│    │    └─BatchNorm2d: 3-24                      [1, 64, 48, 48]           (recursive)
│    │    └─GlobalLocalAttention: 3-25             [1, 64, 48, 48]           (recursive)
│    │    └─Identity: 3-26                         [1, 64, 48, 48]           --
│    │    └─BatchNorm2d: 3-27                      [1, 64, 48, 48]           (recursive)
│    │    └─Mlp: 3-28                              [1, 64, 48, 48]           (recursive)
│    │    └─Identity: 3-29                         [1, 64, 48, 48]           --
│    │    └─ConvBN: 3-30                           [1, 64, 48, 48]           (recursive)
│    │    └─ConvBN: 3-31                           [1, 64, 48, 48]           (recursive)
│    │    └─ConvBN: 3-32                           [1, 64, 48, 48]           (recursive)
│    │    └─SeparableConvBN: 3-33                  [1, 64, 48, 48]           (recursive)
│    └─Fusion: 2-9                                 [1, 64, 96, 96]           --
│    │    └─Conv2d: 3-34                           [1, 64, 96, 96]           4,160
│    │    └─Sequential: 3-35                       [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-36                       [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-37                       [1, 64, 96, 96]           36,992
│    │    └─ConvBNReLU: 3-38                       [1, 64, 96, 96]           (recursive)
│    └─Block: 2-10                                 [1, 64, 96, 96]           (recursive)
│    │    └─BatchNorm2d: 3-39                      [1, 64, 96, 96]           (recursive)
│    │    └─GlobalLocalAttention: 3-40             [1, 64, 96, 96]           (recursive)
│    │    └─Identity: 3-41                         [1, 64, 96, 96]           --
│    │    └─BatchNorm2d: 3-42                      [1, 64, 96, 96]           (recursive)
│    │    └─Mlp: 3-43                              [1, 64, 96, 96]           (recursive)
│    │    └─Identity: 3-44                         [1, 64, 96, 96]           --
│    │    └─ConvBN: 3-45                           [1, 64, 96, 96]           (recursive)
│    │    └─ConvBN: 3-46                           [1, 64, 96, 96]           (recursive)
│    │    └─ConvBN: 3-47                           [1, 64, 96, 96]           (recursive)
│    │    └─SeparableConvBN: 3-48                  [1, 64, 96, 96]           (recursive)
│    └─Fusion: 2-11                                [1, 64, 192, 192]         --
│    │    └─Conv2d: 3-49                           [1, 64, 192, 192]         2,112
│    │    └─Sequential: 3-50                       [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-51                       [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-52                       [1, 64, 192, 192]         36,992
│    │    └─ConvBNReLU: 3-53                       [1, 64, 192, 192]         (recursive)
│    └─FRM: 2-12                                   [1, 6, 768, 768]          --
│    │    └─SeparableConvBN: 3-54                  [1, 64, 192, 192]         4,800
│    │    └─SeparableConvBN: 3-55                  [1, 64, 192, 192]         5,824
│    │    └─ConvBN: 3-56                           [1, 64, 192, 192]         4,224
│    │    └─ConvBNReLU: 3-57                       [1, 64, 192, 192]         36,992
│    │    └─SeparableConvBNReLU: 3-58              [1, 64, 192, 192]         103,104
│    │    └─ConvBN: 3-59                           [1, 64, 192, 192]         (recursive)
│    │    └─ConvBNReLU: 3-60                       [1, 64, 192, 192]         4,224
│    │    └─ConvBNReLU: 3-61                       [1, 64, 192, 192]         (recursive)
│    │    └─Conv2d: 3-62                           [1, 6, 192, 192]          390
====================================================================================================
Total params: 8,908,302
Trainable params: 8,908,302
Non-trainable params: 0
Total mult-adds (G): 19.27
====================================================================================================
Input size (MB): 7.08
Forward/backward pass size (MB): 1360.93
Params size (MB): 35.63
Estimated Total Size (MB): 1403.64
====================================================================================================
```

# 参考
[https://github.com/WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg)
