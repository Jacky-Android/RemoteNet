from functools import partial
from collections import OrderedDict

from .attention import Block


import torch
import torch.nn as nn

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x,H,W


class stage(nn.Module):
    def __init__(self, stride=4, in_chans=3, embed_dim=768,depth=3,drop_path_ratio=0,num_heads=8):
        super().__init__()
        self.path_embeding = OverlapPatchEmbed(embed_dim=embed_dim,stride=stride,in_chans=in_chans)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.Sequential(*[Block(dim=embed_dim,drop_path_ratio=dpr[i],num_heads=num_heads)
                                    for i in range(depth)
                                    ])
        
    def forward(self, x):
        x,H,W = self.path_embeding(x)
        x = self.blocks(x)
        x = x.permute(0,2,1).view(1,x.shape[-1],H,W)
        return x
    
class Encoder(nn.Module):
    def __init__(self, img_size=1024, in_chans=3, depths=(3,4,6,3),dims = (32,64,160,256)):
        super().__init__()
        self.stage1 = stage(embed_dim=dims[0],depth=depths[0],in_chans=in_chans)
        self.stage2 = stage(embed_dim=dims[1],depth=depths[1],in_chans=dims[0],stride=2)
        self.stage3 = stage(embed_dim=dims[2],depth=depths[2],in_chans=dims[1],stride=2)
        self.stage4 = stage(embed_dim=dims[3],depth=depths[3],in_chans=dims[2],stride=2)
        
    def forward(self, x):
        
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage1,stage2,stage3,stage4
    
