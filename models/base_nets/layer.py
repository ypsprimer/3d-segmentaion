import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
# from models.inplace_abn import InPlaceABNSync, InPlaceABN, ABN
from models.inplace_abn import InPlaceABNSync, InPlaceABN, ABN

#######################################################################
############### BasicConv系列，conv + bn + act结构 ####################
#######################################################################
class BasicConv3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConvBN3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConvBN3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = InPlaceABN(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class BasicConvBNS3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConvBNS3d, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = InPlaceABNSync(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


#######################################################################
############### PreConv系列，bn + act + conv结构 ######################
#######################################################################
class PreConv3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConv3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class PreConvBN3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvBN3d, self).__init__()
        self.bn = InPlaceABN(in_planes)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class PreConvBNS3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvBNS3d, self).__init__()
        self.bn = InPlaceABNSync(in_planes)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


######################################################################################
############### BasicConvTranspose反卷积系列，conv + bn + act结构 ####################
######################################################################################
class BasicConvTranspose3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConvTranspose3d, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConvTransposeBN3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConvTransposeBN3d, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = InPlaceABN(out_planes)

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.bn(x)
        return x


class BasicConvTransposeBNS3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConvTransposeBNS3d, self).__init__()
        self.convTranspose = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.bn = InPlaceABNSync(out_planes)

    def forward(self, x):
        x = self.convTranspose(x)
        x = self.bn(x)
        return x


######################################################################################
############### PreConvTranspose反卷积系列，bn + act+ conv结构 #######################
######################################################################################
class PreConvTranspose3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvTranspose3d, self).__init__()
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class PreConvTransposeBN3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvTransposeBN3d, self).__init__()
        self.bn = InPlaceABN(in_planes)
        self.conv = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class PreConvTransposeBNS3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvTransposeBNS3d, self).__init__()
        self.bn = InPlaceABNSync(in_planes)
        self.conv = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


###############################################################################
############### DownBlock降采样，conv + bn + act结构 ##########################
###############################################################################
class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownBlockBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(DownBlockBN, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = InPlaceABN(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DownBlockBNS(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(DownBlockBNS, self).__init__()
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = InPlaceABNSync(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


###############################################################################
############### PreDownBlock降采样，bn + act+ conv结构 ###########################
###############################################################################
class PreDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(PreDownBlock, self).__init__()
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return x


class PreDownBlockBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(PreDownBlockABN, self).__init__()
        self.bn = InPlaceABN(in_planes)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


class PreDownBlockBNS(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(PreDownBlockBNS, self).__init__()
        self.bn = InPlaceABNSync(in_planes)
        self.conv = nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        return x


##################################################
############### 单卷积结构 #######################
##################################################


def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias
    )


def conv5x5x5(in_planes, out_planes, stride=1, bias=False):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=bias
    )


####################################################
############### Resblock结构 #######################
####################################################
class ResBlockBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlockBN, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = InPlaceABNSync(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = InPlaceABNSync(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResBlockGN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlockGN, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(4, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.GroupNorm(4, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


####################################################
############### Bottleneck结构 #####################
####################################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = InPlaceABNSync(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = InPlaceABNSync(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = InPlaceABNSync(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


###############################################################
############### Pre inplace堆叠卷积结构 #######################
###############################################################
class Basic_block_inplace(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(Basic_block_inplace, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=True)
        self.bn1 = InPlaceABNSync(planes)
        self.conv2 = conv3x3x3(planes, planes, bias=True)
        self.bn2 = InPlaceABNSync(planes)
        self.stride = stride

    def forward(self, x):
        xcopy = x.clone()
        out = self.bn1(xcopy)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        return out + x


class Bottleneck_inplace(nn.Module):  #### 利用1*1卷积缩小/放大信道数目
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, abn=InPlaceABN):
        super(Bottleneck_inplace, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = abn(inplanes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = abn(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = abn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = x.clone()
        out = self.bn1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.conv3(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out
