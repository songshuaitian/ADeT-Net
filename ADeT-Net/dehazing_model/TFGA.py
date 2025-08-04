import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL.ImageOps import expand

from .CFFB import CFFB


class CrossAttention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CrossAttention, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1, stride=1),
                                   # nn.InstanceNorm2d(in_channel),
                                   nn.ReLU(),
                                   )
    def forward(self, f1, f2):
        f1_hat = f1
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        att_map = f1 * f2
        att_shape = att_map.shape
        att_map = torch.reshape(att_map, [att_shape[0], att_shape[1], -1])
        att_map = F.softmax(att_map, dim=2)
        att_map = torch.reshape(att_map, att_shape)
        f1 = f1 * att_map
        f1 = f1 + f1_hat
        return f1
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
class UP(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class TFGA(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(TFGA,self).__init__()
        self.UP1 = UP(
            patch_size=2, out_chans=48, embed_dim=96)
        self.UP2 = UP(
            patch_size=2, out_chans=24, embed_dim=48)
        self.conv_up = nn.Conv2d(24,dim_in,1,1)
        self.crossatt1 = CrossAttention(dim_in,dim_out)
        self.crossatt1_2 = CrossAttention(dim_in,dim_out)
        self.crossatt2 = CrossAttention(dim_in,dim_out)
        self.crossatt2_1 = CrossAttention(dim_in, dim_out)
        self.conv1 = nn.Conv2d(2*dim_in,dim_in,1,1)
        self.convnext = nn.Sequential(
            CFFB(dim_in),
            CFFB(dim_in),
        )
        self.conv2= nn.Conv2d(2*dim_in,dim_in,1,1)
        self.conv3= nn.Conv2d(2*dim_in,dim_in,1,1)
        self.weight = weight(dim_in)
        self.weight1 = weight(dim_in)
        self.conv4 = nn.Conv2d(dim_in,dim_in,3,1,1)
        self.conv5 = nn.Conv2d(2*dim_in,dim_in,1,1)
    def forward(self,feature1,feature2,feature3):

        feature3 = self.UP1(feature3)
        feature3 = self.UP2(feature3)
        feature3 = self.conv_up(feature3)
        feature_cat = torch.cat([feature1,feature2],dim=1)
        feature_cat = self.conv1(feature_cat)
        feature_cat1 = self.crossatt1(feature1,feature_cat)
        feature_cat1_2 = self.crossatt1_2(feature_cat,feature1)



        feature_cat2 = self.crossatt2(feature2,feature_cat)
        feature_cat2_1 = self.crossatt2(feature_cat,feature2)
        feature_cat1_3 = torch.cat([feature_cat1, feature_cat2],dim=1)
        feature_cat1_3 = self.conv2(feature_cat1_3)
        feature_cat2_4 = torch.cat([feature_cat1_2,feature_cat2_1],dim=1)
        feature_cat2_4 = self.conv3(feature_cat2_4)


        feature = feature_cat2_4 + feature_cat1_3

        feature = self.convnext(feature)
        feature_weight =self.weight(feature)
        feature_weight1 =self.weight1(feature)
        feature1 = feature1 * feature_weight
        feature2 = feature2 * feature_weight1

        feature = feature + feature1 + feature2
        feature = self.conv4(feature)
        feature = torch.cat([feature3,feature],dim=1)
        feature = self.conv5(feature)
        return feature


if __name__ == '__main__':
    feature1 = torch.randn(4,3,256,256)
    feature2 = torch.randn(4,3,256,256)
    model = TFGA(3,3)
    res = model(feature1,feature2)

    print(res.shape)
