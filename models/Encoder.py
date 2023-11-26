from functools import partial
from collections import OrderedDict

from .attention import Block
import os
os.environ['CURL_CA_BUNDLE'] = ''


import torch
import torch.nn as nn

import torch
import torch.nn as nn
import timm


    
class Encoder(nn.Module):
    def __init__(self, encoder='pvt_v2_b1'):
        super().__init__()
        self.Encoder = timm.create_model(encoder,pretrained=True,features_only=True,out_indices=(0,1,2,3))
        
    def forward(self, x):
        
        stage1,stage2,stage3,stage4 = self.Encoder(x)
        return stage1,stage2,stage3,stage4
    
