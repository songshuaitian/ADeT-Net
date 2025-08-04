import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL.ImageOps import expand
from .CFFB import CFFB

class text_MLP(nn.Module):
    def __init__(self,text_dim,img_dim):
        super(text_MLP, self).__init__()
        self.conv1 = nn.Conv2d(text_dim,text_dim,1,1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(text_dim,img_dim,1,1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
class MLP(nn.Module):
    def __init__(self,dim):
        super(MLP, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1,1)
        )
    def forward(self,x):
        x1 = self.conv(x)
        return x + x1
class weight(nn.Module):
    def __init__(self,dim):
        super(weight, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1,1),
        )
    def forward(self,x):
        x1 = self.conv(x)
        return F.softmax(x1,dim=1)
class IGM(nn.Module):
    def __init__(self,dim,text_dim):
        super(IGM,self).__init__()
        self.dim = dim
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.text_mlp = text_MLP(text_dim,dim)
        self.mlp1 = MLP(dim)
        self.mlp2 = MLP(dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(dim*2), dim, 1),
            nn.ReLU(inplace=True)
            )
        self.softmax = weight(dim)
        self.convnext = CFFB(dim)
    def forward(self,feature,text):
        B,C,H,W = feature.shape
        text = text.unsqueeze(2).unsqueeze(3)
        text = text.expand(B,-1,-1,-1)#B,128,1,1
        text = self.text_mlp(text)#B,96,1,1
        feature_max = self.maxpool(self.mlp1(feature))#B,96,1,1
        feature_avg = self.avgpool(self.mlp2(feature))
        feature_text = torch.cat([(feature_avg + feature_max),text],dim=1)#B,192,1,1
        feature_text = self.conv1(feature_text)#B,96,1,1
        feature_text_w = self.softmax(feature_text)#B,96,1,1
        feature_text_w = feature_text_w.expand(-1, -1, H, W)  # B,96,H,W
        feature_w = feature_text_w * feature#B,96,1,1
        feature_w = self.convnext(feature_w) + feature
        return feature_w


if __name__ == '__main__':
    text = torch.randn(1,128)
    img = torch.randn(4,48,112,94)
    model = IGM(48,128)
    res = model(img,text)

    print(res.shape,1111)
