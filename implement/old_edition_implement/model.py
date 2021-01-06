import torch
import torch.nn as nn
import torch.nn.functional as F
from inplace_abn import InPlaceABN, InPlaceABNSync


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
        self.bn = InPlaceABNSync(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class PreConvBN3d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(PreConvBN3d, self).__init__()
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


class SingleLayerBN(nn.Module):
    def __init__(self, inChannels, growth_Channels):
        super(SingleLayerBN, self).__init__()
        self.bn = InPlaceABNSync(inChannels)
        self.conv = nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = torch.cat((x, out), 1)
        return out


class Dense_Vnet1(nn.Module):
    def __init__(self):
        super(Dense_Vnet1, self).__init__()
        self.downblock1 = DownBlockBN(1, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=4
        )
        self.downblock2 = DownBlockBN(128, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock2 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=8
        )
        self.downblock3 = DownBlockBN(256, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock3 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=12
        )
        self.skipconv1 = nn.Conv3d(128, 24, kernel_size=3, padding=1)
        self.skipconv2 = nn.Conv3d(256, 24, kernel_size=3, padding=1)
        self.skipconv3 = nn.Conv3d(384, 24, kernel_size=3, padding=1)
        self.FCN_out = PreConvBN3d(72, 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(1)

    def _make_denseBN(self, inChannels, growth_Channels, nDenseLayers):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(SingleLayerBN(inChannels, growth_Channels))
                inChannels += growth_Channels
        layers.append(InPlaceABNSync(inChannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downblock1(x)
        x = self.denseblock1(x)
        x1 = self.downblock2(x)
        x1 = self.denseblock2(x1)
        x2 = self.downblock3(x1)
        x2 = self.denseblock3(x2)
        x = F.interpolate(self.skipconv1(x), scale_factor=2, mode="trilinear")
        x1 = F.interpolate(self.skipconv2(x1), scale_factor=4, mode="trilinear")
        x2 = F.interpolate(self.skipconv3(x2), scale_factor=8, mode="trilinear")
        #         align_corners=True
        out = self.FCN_out(torch.cat((x, x1, x2), 1))
        return self.softmax(out)


class Dense_Vnet2(nn.Module):
    def __init__(self):
        super(Dense_Vnet2, self).__init__()
        self.downblock1 = DownBlockBN(1, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=4
        )
        self.downblock2 = DownBlockBN(128, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock2 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=8
        )
        self.downblock3 = DownBlockBN(256, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock3 = self._make_denseBN(
            inChannels=32, growth_Channels=32, nDenseLayers=12
        )
        self.skipconv1 = nn.Conv3d(128, 24, kernel_size=3, padding=1)
        self.skipconv2 = nn.Conv3d(256, 24, kernel_size=3, padding=1)
        self.skipconv3 = nn.Conv3d(384, 24, kernel_size=3, padding=1)
        self.FCN_out = PreConvBN3d(72, 1, kernel_size=3, padding=1)

    def _make_denseBN(self, inChannels, growth_Channels, nDenseLayers):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(SingleLayerBN(inChannels, growth_Channels))
                inChannels += growth_Channels
        layers.append(InPlaceABNSync(inChannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downblock1(x)
        x = self.denseblock1(x)
        x1 = self.downblock2(x)
        x1 = self.denseblock2(x1)
        x2 = self.downblock3(x1)
        x2 = self.denseblock3(x2)
        x = F.interpolate(self.skipconv1(x), scale_factor=2, mode="trilinear")
        x1 = F.interpolate(self.skipconv2(x1), scale_factor=4, mode="trilinear")
        x2 = F.interpolate(self.skipconv3(x2), scale_factor=8, mode="trilinear")
        #         align_corners=True
        out = self.FCN_out(torch.cat((x, x1, x2), 1))
        return out
