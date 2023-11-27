import torch
import torch.nn as nn
import torch.nn.functional as F
from .CNNs import ConvBN, SeparableConvBN, Conv,SeparableConvBN,ConvBNReLU,SeparableConvBNReLU
from timm.layers.drop import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        
        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.ws)
        coords_w = torch.arange(self.ws)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.ws - 1
        relative_coords[:, :, 0] *= 2 * self.ws - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)


    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x
    
    def forward(self,x):
        B, C, H, W = x.shape

        local  = self.local1(x)+self.local2(x)
        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        
        qkv = self.qkv(x)
        qkv = qkv.contiguous().view(B, Hp // self.ws, self.ws, Wp // self.ws, self.ws, 3*C).permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.ws, self.ws, 3*C)
        q, k, v = qkv.reshape(qkv.shape[0], self.ws*self.ws,3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        dots += relative_position_bias.unsqueeze(0)
        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = attn.reshape(B, C, H,W)[:, :, :H, :W]
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=64, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.cnn1 = ConvBN(dim,dim,1)
        self.cnn2 = ConvBN(dim,dim,3)
        self.cnn3 = ConvBN(dim,dim,5)
        self.output = SeparableConvBN(dim,dim,7)

    def forward(self, x):
        shotcut = x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.cnn1(shotcut)+self.cnn2(shotcut)+self.cnn3(shotcut)+x
        
        x = self.output(x)
        return x
    
class Fusion(nn.Module):
    def __init__(self, in_channels=128, decode_channels=64):
        super(Fusion, self).__init__()
        self.preconv = nn.Conv2d(in_channels, decode_channels, kernel_size=1, stride=1, padding=0)
        self.attention_map = nn.Sequential(nn.BatchNorm2d(num_features=decode_channels),
                                           nn.LeakyReLU(),
                                           nn.MaxPool2d(kernel_size=2,stride=2),
                                           nn.Conv2d(decode_channels, decode_channels, kernel_size=3, stride=1),
                                           nn.BatchNorm2d(decode_channels),
                                           nn.LeakyReLU(),
                                           nn.AdaptiveAvgPool2d(1),
                                           nn.Conv2d(decode_channels, decode_channels, kernel_size=1),
                                           nn.BatchNorm2d(num_features=decode_channels),
                                           nn.Sigmoid()
                                           )
        self.CNN = ConvBNReLU(decode_channels,decode_channels)

    def forward(self, x, stage):
        stage = self.preconv(stage)
        stage = self.attention_map(stage)*stage

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.attention_map(x)*x+stage
        x = self.CNN(self.CNN(x))+x
        return x

class FRM(nn.Module):
    def __init__(self, dim = 64,num_classes=6):
        super().__init__()
        self.cnn1 = SeparableConvBN(dim,dim,3)
        self.cnn2 = SeparableConvBN(dim,dim,5)
        self.cnn3 = SeparableConvBNReLU(dim,dim,3)
        self.cnn4 = ConvBN(dim,dim,1)
        self.cnn5 = ConvBNReLU(dim,dim,3)
        self.cnn6 = ConvBNReLU(dim,dim,1)
        self.head = nn.Conv2d(dim,num_classes,1)
        

    
    def forward(self, x):
        shotcut = x
        x = self.cnn5(self.cnn4(self.cnn1(x)+self.cnn2(x)))+self.cnn3(x)
        x = self.cnn4(shotcut)+x
        x = x*self.cnn6(x)
        x = self.head(self.cnn5(x))
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

        return x


class Decoder(nn.Module):
    def __init__(self, dim = 64,dims = (32,64,160,256),num_classes=6):
        super().__init__()
        self.preconv = nn.Conv2d(dims[-1], dim, kernel_size=1, stride=1, padding=0)
        self.block4 = Block(dim=dim, num_heads=8, mlp_ratio=4, qkv_bias=True)
        self.F1 = Fusion(in_channels=dims[-2],decode_channels=dim)
        self.block3 = Block(dim=dim, num_heads=8, mlp_ratio=4, qkv_bias=True)
        self.F2 = Fusion(in_channels=dims[-3],decode_channels=dim)
        self.block2 = Block(dim=dim, num_heads=8, mlp_ratio=4, qkv_bias=True)
        self.F3 = Fusion(in_channels=dims[-4],decode_channels=dim)
        self.FRM = FRM(dim,num_classes=num_classes)

    def forward(self, stage1,stage2,stage3,stage4):

        x = self.preconv(stage4)
        x = self.block4(x)
        x = self.F1(x,stage3)
        x = self.block3(x)
        x = self.F2(x,stage2)
        x = self.block2(x)
        x = self.F3(x,stage1)

        x = self.FRM(x)
        return x



   
