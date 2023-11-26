# RemoteNet: Remote Sensing Image Segmentation Network based on Global-Local Information
Paper:[RemoteNet: Remote Sensing Image Segmentation Network based on Global-Local Information](https://arxiv.org/abs/2302.13084)

Because the original paper does not have open source code, I tried to reproduce it.
## This project reproduces this paper
# Major changes
😒😒😒 Although the paper does not say it, the Encoder is [pvt_v2](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/pvt_v2.py),so the code was modified!!!😢😢

# Environment
```python
torch==2.1.1+cu118
torchaudio==2.1.1+cu118
torchinfo==1.8.0
torchvision==0.16.1
requests==2.28.1
urllib3== 1.25.11
timm==0.9.0 #It must be ensured that timm is this version!!!
```
# Model summary

```python
==============================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
==============================================================================================================
RemoteNet                                                    [1, 6, 1024, 1024]        --
├─Encoder: 1-1                                               [1, 64, 256, 256]         --
│    └─FeatureListNet: 2-1                                   [1, 64, 256, 256]         --
│    │    └─OverlapPatchEmbed: 3-1                           [1, 256, 256, 64]         9,600
│    │    └─PyramidVisionTransformerStage: 3-2               [1, 64, 256, 256]         701,056
│    │    └─PyramidVisionTransformerStage: 3-3               [1, 128, 128, 128]        1,279,616
│    │    └─PyramidVisionTransformerStage: 3-4               [1, 320, 64, 64]          3,682,880
│    │    └─PyramidVisionTransformerStage: 3-5               [1, 512, 32, 32]          7,822,848
├─Decoder: 1-2                                               [1, 6, 1024, 1024]        --
│    └─Conv2d: 2-2                                           [1, 64, 32, 32]           32,832
│    └─Block: 2-3                                            [1, 64, 32, 32]           --
│    │    └─BatchNorm2d: 3-6                                 [1, 64, 32, 32]           128
│    │    └─GlobalLocalAttention: 3-7                        [1, 64, 32, 32]           63,816
│    │    └─Identity: 3-8                                    [1, 64, 32, 32]           --
│    │    └─BatchNorm2d: 3-9                                 [1, 64, 32, 32]           128
│    │    └─Mlp: 3-10                                        [1, 64, 32, 32]           33,088
│    │    └─Identity: 3-11                                   [1, 64, 32, 32]           --
│    │    └─ConvBN: 3-12                                     [1, 64, 32, 32]           4,224
│    │    └─ConvBN: 3-13                                     [1, 64, 32, 32]           36,992
│    │    └─ConvBN: 3-14                                     [1, 64, 32, 32]           102,528
│    │    └─SeparableConvBN: 3-15                            [1, 64, 32, 32]           7,360
│    └─Fusion: 2-4                                           [1, 64, 64, 64]           --
│    │    └─Conv2d: 3-16                                     [1, 64, 64, 64]           20,544
│    │    └─Sequential: 3-17                                 [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-18                                 [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-19                                 [1, 64, 64, 64]           36,992
│    │    └─ConvBNReLU: 3-20                                 [1, 64, 64, 64]           (recursive)
│    └─Block: 2-5                                            [1, 64, 64, 64]           --
│    │    └─BatchNorm2d: 3-21                                [1, 64, 64, 64]           128
│    │    └─GlobalLocalAttention: 3-22                       [1, 64, 64, 64]           63,816
│    │    └─Identity: 3-23                                   [1, 64, 64, 64]           --
│    │    └─BatchNorm2d: 3-24                                [1, 64, 64, 64]           128
│    │    └─Mlp: 3-25                                        [1, 64, 64, 64]           33,088
│    │    └─Identity: 3-26                                   [1, 64, 64, 64]           --
│    │    └─ConvBN: 3-27                                     [1, 64, 64, 64]           4,224
│    │    └─ConvBN: 3-28                                     [1, 64, 64, 64]           36,992
│    │    └─ConvBN: 3-29                                     [1, 64, 64, 64]           102,528
│    │    └─SeparableConvBN: 3-30                            [1, 64, 64, 64]           7,360
│    └─Fusion: 2-6                                           [1, 64, 128, 128]         --
│    │    └─Conv2d: 3-31                                     [1, 64, 128, 128]         8,256
│    │    └─Sequential: 3-32                                 [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-33                                 [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-34                                 [1, 64, 128, 128]         36,992
│    │    └─ConvBNReLU: 3-35                                 [1, 64, 128, 128]         (recursive)
│    └─Block: 2-7                                            [1, 64, 128, 128]         --
│    │    └─BatchNorm2d: 3-36                                [1, 64, 128, 128]         128
│    │    └─GlobalLocalAttention: 3-37                       [1, 64, 128, 128]         63,816
│    │    └─Identity: 3-38                                   [1, 64, 128, 128]         --
│    │    └─BatchNorm2d: 3-39                                [1, 64, 128, 128]         128
│    │    └─Mlp: 3-40                                        [1, 64, 128, 128]         33,088
│    │    └─Identity: 3-41                                   [1, 64, 128, 128]         --
│    │    └─ConvBN: 3-42                                     [1, 64, 128, 128]         4,224
│    │    └─ConvBN: 3-43                                     [1, 64, 128, 128]         36,992
│    │    └─ConvBN: 3-44                                     [1, 64, 128, 128]         102,528
│    │    └─SeparableConvBN: 3-45                            [1, 64, 128, 128]         7,360
│    └─Fusion: 2-8                                           [1, 64, 256, 256]         --
│    │    └─Conv2d: 3-46                                     [1, 64, 256, 256]         4,160
│    │    └─Sequential: 3-47                                 [1, 64, 1, 1]             41,344
│    │    └─Sequential: 3-48                                 [1, 64, 1, 1]             (recursive)
│    │    └─ConvBNReLU: 3-49                                 [1, 64, 256, 256]         36,992
│    │    └─ConvBNReLU: 3-50                                 [1, 64, 256, 256]         (recursive)
│    └─FRM: 2-9                                              [1, 6, 1024, 1024]        --
│    │    └─SeparableConvBN: 3-51                            [1, 64, 256, 256]         4,800
│    │    └─SeparableConvBN: 3-52                            [1, 64, 256, 256]         5,824
│    │    └─ConvBN: 3-53                                     [1, 64, 256, 256]         4,224
│    │    └─ConvBNReLU: 3-54                                 [1, 64, 256, 256]         36,992
│    │    └─SeparableConvBNReLU: 3-55                        [1, 64, 256, 256]         103,104
│    │    └─ConvBN: 3-56                                     [1, 64, 256, 256]         (recursive)
│    │    └─ConvBNReLU: 3-57                                 [1, 64, 256, 256]         4,224
│    │    └─ConvBNReLU: 3-58                                 [1, 64, 256, 256]         (recursive)
│    │    └─Conv2d: 3-59                                     [1, 6, 256, 256]          390
==============================================================================================================
Total params: 14,701,150
Trainable params: 14,701,150
Non-trainable params: 0
Total mult-adds (G): 34.69
==============================================================================================================
Input size (MB): 12.58
Forward/backward pass size (MB): 4018.96
Params size (MB): 58.78
Estimated Total Size (MB): 4090.32
==============================================================================================================
```

# 参考
[https://github.com/WangLibo1995/GeoSeg](https://github.com/WangLibo1995/GeoSeg)
