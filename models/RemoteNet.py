import torch
import torch.nn as nn
import torch.nn.functional as F

from .Decoder import Decoder
from .Encoder import Encoder

class RemoteNet(nn.Module):
    def __init__(self, dim = 64,dims = (32,64,160,256),num_classes=6):
        super().__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder(dim=dim,dims=dims,num_classes=num_classes)
    def forward(self, x):
        stage1,stage2,stage3,stage4  = self.Encoder(x)
        x = self.Decoder(stage1,stage2,stage3,stage4)
        return x
