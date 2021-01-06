import torch
from torch import nn
import torch.nn.functional as F
import copy

import sys
from sys import path
# path.append(sys.path[0])
# print(path)
# from inplace_abn import InPlaceABN, InPlaceABNSync, ABN
# from base_nets.layer import Bottleneck_inplace
from .inplace_abn import InPlaceABN, InPlaceABNSync, ABN
from .base_nets.layer import Bottleneck_inplace

from collections import OrderedDict

class ResBlock(nn.Module):
    def __init__(self, inplanes, planes, abn, stride=1, downsample=None, ):
        super(ResBlock, self).__init__()

        
        self.conv1 = nn.Conv3d(inplanes, planes, stride=stride, kernel_size=3, padding=1)
        self.conv1_bn = abn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, padding=1)
        self.conv2_bn = abn(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        
        out = x.clone()

        out = self.conv1(out)
        out = self.conv1_bn(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)

        if self.downsample is not None:
            x = self.downsample(x)
        
        # 保持正确的梯度流动：x -> out -> x + out 
        out = out + x
        out = self.relu(out)

        return out
 


class multiClass_unet_residualBlock(nn.Module):
    def __init__(self, n_inp=1, n_out=1, feats=(32, 32, 64, 64, 128, 128, 128), abn=2, n_encoders=2,):
        """
        :param n_inp: 输入channel数量，3D图像默认为1
        :param n_out: 输出channel数量，与n_inp一致
        :param feats: 经过conv后的channel数量
        :param abn: [0,1,2]指定类型的abn
        :param n_encoders:

        """
        super().__init__()

        self.previous_encoder = 0
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.n_inp = n_inp
        if abn == 0:
            abnblock = ABN
        elif abn == 1:
            abnblock = InPlaceABN
        elif abn == 2:
            abnblock = InPlaceABNSync

        # input layer
        # stride = 1
        self.in_layer = nn.Sequential(
            nn.Conv3d(n_inp, feats[0], kernel_size=3, padding=1),
            abnblock(feats[0]),
            nn.ReLU(),
        )
        self.in_layer_resbk = ResBlock(inplanes=feats[0], planes=feats[0], downsample=None, abn=abnblock)

        # down layer1
        # stride = [1,2,2]
        self.lconvlayer1 = ResBlock(inplanes=feats[0], planes=feats[1], abn=abnblock, stride=(1,2,2), 
                                    downsample=nn.Conv3d(in_channels=feats[0], out_channels=feats[1], stride=(1,2,2), kernel_size=3, padding=1),
                                    )
        self.lconvlayer1_resbk = ResBlock(inplanes=feats[1], planes=feats[1],abn=abnblock)

        # down layer2
        # stride = [1,2,2]
        self.lconvlayer2 = ResBlock(inplanes=feats[1], planes=feats[2], abn=abnblock, stride=(1,2,2), 
                                    downsample=nn.Conv3d(in_channels=feats[1], out_channels=feats[2], stride=(1,2,2), kernel_size=3, padding=1),
                                    )

        self.lconvlayer2_resbk = nn.Sequential()
        for i in range(2):
            self.lconvlayer2_resbk.add_module('resblock_{}'.format(i), ResBlock(inplanes=feats[2], planes=feats[2], abn=abnblock))
           
        # down layer3
        # stride = [1,2,2]
        self.lconvlayer3 = ResBlock(inplanes=feats[2], planes=feats[3], abn=abnblock, stride=(1,2,2), 
                                    downsample=nn.Conv3d(in_channels=feats[2], out_channels=feats[3], stride=(1,2,2), kernel_size=3, padding=1),
                                    )

        self.lconvlayer3_resbk = nn.Sequential()
        for i in range(2):
            self.lconvlayer3_resbk.add_module('resblock_{}'.format(i), ResBlock(inplanes=feats[3], planes=feats[3], abn=abnblock))
        

        # down layer4
        # stride = 2
        self.lconvlayer4 = ResBlock(inplanes=feats[3], planes=feats[4], abn=abnblock, stride=2, 
                                    downsample=nn.Conv3d(in_channels=feats[3], out_channels=feats[4], stride=2, kernel_size=3, padding=1),
                                    )

        self.lconvlayer4_resbk = nn.Sequential()
        for i in range(3):
            self.lconvlayer4_resbk.add_module('resblock_{}'.format(i), ResBlock(inplanes=feats[4], planes=feats[4], abn=abnblock))
    
        
        # down layer5
        # stride = 2
        self.lconvlayer5 = ResBlock(inplanes=feats[4], planes=feats[5], abn=abnblock, stride=2, 
                                    downsample=nn.Conv3d(in_channels=feats[4], out_channels=feats[5], stride=2, kernel_size=3, padding=1),
                                    )

        self.lconvlayer5_resbk = nn.Sequential()
        for i in range(3):
            self.lconvlayer5_resbk.add_module('resblock_{}'.format(i), ResBlock(inplanes=feats[5], planes=feats[5], abn=abnblock))
    
        
        # up layer5
        self.rconvlayer5 = nn.Sequential(
            self.upsample,
            nn.Conv3d(in_channels=feats[5], out_channels=feats[4], kernel_size=3, padding=1),
            abnblock(feats[4]),
            nn.ReLU(),
        )
        self.rconvlayer5_resbk = nn.Sequential(
            nn.Conv3d(in_channels=feats[4], out_channels=feats[4], kernel_size=3, padding=1),
            abnblock(feats[4]),
            nn.ReLU(),
        )

        # up layer4
        self.rconvlayer4 = nn.Sequential(
            self.upsample,
            nn.Conv3d(in_channels=feats[4], out_channels=feats[3], kernel_size=3, padding=1),
            abnblock(feats[3]),
            nn.ReLU(),
        )
        self.rconvlayer4_resbk = nn.Sequential(
            nn.Conv3d(in_channels=feats[3], out_channels=feats[3], kernel_size=3, padding=1),
            abnblock(feats[3]),
            nn.ReLU(),
        )

        # up layer3
        self.rconvlayer3 = nn.Sequential(
            self.upsamplez,
            nn.Conv3d(in_channels=feats[3], out_channels=feats[2], kernel_size=3, padding=1),
            abnblock(feats[2]),
            nn.ReLU(),
        )
        self.rconvlayer3_resbk = nn.Sequential(
            nn.Conv3d(in_channels=feats[2], out_channels=feats[2], kernel_size=3, padding=1),
            abnblock(feats[2]),
            nn.ReLU(),
        )

        # up layer2
        self.rconvlayer2 = nn.Sequential(
            self.upsamplez,
            nn.Conv3d(in_channels=feats[2], out_channels=feats[1], kernel_size=3, padding=1),
            abnblock(feats[1]),
            nn.ReLU(),
        )
        self.rconvlayer2_resbk = nn.Sequential(
            nn.Conv3d(in_channels=feats[1], out_channels=feats[1], kernel_size=3, padding=1),
            abnblock(feats[1]),
            nn.ReLU(),
        )

        # up layer1
        self.rconvlayer1 = nn.Sequential(
            self.upsamplez,
            nn.Conv3d(in_channels=feats[1], out_channels=feats[0], kernel_size=3, padding=1),
            abnblock(feats[0]),
            nn.ReLU(),
        )
        self.rconvlayer1_resbk = nn.Sequential(
            nn.Conv3d(in_channels=feats[0], out_channels=feats[0], kernel_size=3, padding=1),
            abnblock(feats[0]),
            nn.ReLU(),
        )

        # out layer
        self.out_layer_0 = nn.Conv3d(feats[0], n_out, kernel_size=1, stride=1)
        self.out_layer_1 = nn.Conv3d(feats[0], n_out, kernel_size=1, stride=1)
        self.out_layer_2 = nn.Conv3d(feats[0], n_out, kernel_size=1, stride=1)


    
    def forward(self, x, *args):
        xl0 = self.in_layer(x)
        xl0 = self.in_layer_resbk(xl0)

        xl1 = self.lconvlayer1(xl0)
        xl1 = self.lconvlayer1_resbk(xl1)

        xl2 = self.lconvlayer2(xl1)
        xl2 = self.lconvlayer2_resbk(xl2)

        xl3 = self.lconvlayer3(xl2)
        xl3 = self.lconvlayer3_resbk(xl3)

        xl4 = self.lconvlayer4(xl3)
        xl4 = self.lconvlayer4_resbk(xl4)

        xl5 = self.lconvlayer5(xl4)
        xl5 = self.lconvlayer5_resbk(xl5)

        # # upsample -> skip connect -> residual block
        xr5 = xl5

        xr4_r = self.rconvlayer5(xr5)
        xr4 = torch.add(xr4_r, xl4)
        xr4 = self.rconvlayer5_resbk(xr4)
        
        xr3_r = self.rconvlayer4(xr4)
        xr3 = torch.add(xr3_r, xl3)
        xr3 = self.rconvlayer4_resbk(xr3)

        xr2_r = self.rconvlayer3(xr3)
        xr2 = torch.add(xr2_r, xl2)
        xr2 = self.rconvlayer3_resbk(xr2)

        xr1_r = self.rconvlayer2(xr2)
        xr1 = torch.add(xr1_r, xl1)
        xr1 = self.rconvlayer2_resbk(xr1)

        xr0_r = self.rconvlayer1(xr1)
        xr0 = torch.add(xr0_r, xl0)
        xr0 = self.rconvlayer1_resbk(xr0)

        xr0_0 = self.out_layer_0(xr0)
        xr0_1 = self.out_layer_1(xr0)
        xr0_2 = self.out_layer_2(xr0)
        out_layer = torch.cat((xr0_0, xr0_1, xr0_2), dim=1) # [bs, class_num(3), z, y, x]

        return F.softmax(out_layer, dim=1)




# MICCAI-2019 Kidney and Kidney Tumor Segmentation Challenge Top-1st
class multi_class_unet_residual_block(multiClass_unet_residualBlock):
    def __init__(self, abn=2):
        super().__init__(
            n_inp=1, 
            n_out=1, 
            feats=(32, 32, 64, 64, 128, 128, 128), 
            abn=abn, 
            n_encoders=2,
        )


if __name__ == "__main__":
    inputs = torch.randn((1, 1, 10, 20, 20))
    # net = ResBlock(inplanes=1, planes=16, stride=2, downsample=nn.Conv3d(in_channels=1, out_channels=16, stride=2, kernel_size=3, padding=1))
    # outputs = net(inputs)

    # print(outputs.shape)
    net = multi_class_unet_residual_block()
    outputs = net(inputs)
    print(outputs.shape)