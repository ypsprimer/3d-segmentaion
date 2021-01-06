import torch
import torch.nn.functional as F
from torch import nn

from inplace_abn import InPlaceABNSync


class Dense_Vnet_2(nn.Module):
    def __init__(self):
        super(Dense_Vnet_2, self).__init__()
        self.downblock1 = DownBlockBN(2, 32, kernel_size=3, stride=2, padding=1)
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
        x = F.interpolate(
            self.skipconv1(x), scale_factor=2, mode="trilinear", align_corners=True
        )
        x1 = F.interpolate(
            self.skipconv2(x1), scale_factor=4, mode="trilinear", align_corners=True
        )
        x2 = F.interpolate(
            self.skipconv3(x2), scale_factor=8, mode="trilinear", align_corners=True
        )
        out = self.FCN_out(torch.cat((x, x1, x2), 1))
        return out


class Dense_Vnet_3(nn.Module):
    def __init__(self):
        super(Dense_Vnet_3, self).__init__()

        self.downblock1 = DownBlockBN(1, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock1 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=4,
            z_dilation=(7, 5, 2, 1),
            z_padding=(7, 5, 2, 1),
        )

        self.downblock2 = DownBlockBN(128, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock2 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=8,
            z_dilation=(5, 2, 1, 1),
            z_padding=(5, 2, 1, 1),
        )

        self.downblock3 = DownBlockBN(256, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock3 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=10,
            z_dilation=(2, 1, 1, 1),
            z_padding=(2, 1, 1, 1),
        )

        self.skipconv1 = nn.Conv3d(192, 16, kernel_size=3, padding=1)
        self.skipconv2 = nn.Conv3d(256, 16, kernel_size=3, padding=1)
        self.skipconv3 = nn.Conv3d(320, 32, kernel_size=3, padding=1)

        self.FCN_out = PreConvBN3d(64, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(1)
        print("finish building")

    def _make_denseBN(
        self,
        inChannels,
        growth_Channels,
        nDenseLayers,
        z_dilation=(7, 5, 2, 1),
        z_padding=(7, 5, 2, 1),
    ):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayerBN(
                        inChannels,
                        growth_Channels,
                        z_dilation=z_dilation[i % len(z_dilation)],
                        z_padding=z_padding[i % len(z_dilation)],
                    )
                )
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
        x = F.interpolate(
            self.skipconv1(x), scale_factor=2, mode="trilinear", align_corners=True
        )
        x1 = F.interpolate(
            self.skipconv2(x1), scale_factor=4, mode="trilinear", align_corners=True
        )
        x2 = F.interpolate(
            self.skipconv3(x2), scale_factor=8, mode="trilinear", align_corners=True
        )
        out = self.FCN_out(torch.cat((x, x1, x2), 1))
        return out


class Dense_Vnet_4(nn.Module):
    def __init__(self):
        super(Dense_Vnet_4, self).__init__()
        self.downblock1 = DownBlockBN(1, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock1 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=4,
            z_dilation=(1, 1, 1, 1),
            z_padding=(1, 1, 1, 1),
        )

        self.downblock2 = DownBlockBN(128, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock2 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=8,
            z_dilation=(1, 1, 1, 1),
            z_padding=(1, 1, 1, 1),
        )

        self.downblock3 = DownBlockBN(256, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock3 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=10,
            z_dilation=(1, 1, 1, 1),
            z_padding=(1, 1, 1, 1),
        )

        self.upblock1 = UpBlockBN(192, 64, kernel_size=4, stride=2, padding=1)
        self.upblock2 = UpBlockBN(320, 64, kernel_size=4, stride=2, padding=1)
        self.upblock3 = UpBlockBN(320, 64, kernel_size=4, stride=2, padding=1)

        self.FCN_out = PreConvBN3d(64, 1, kernel_size=3, padding=1)

        print("finish building")

    def _make_denseBN(
        self,
        inChannels,
        growth_Channels,
        nDenseLayers,
        z_dilation=(7, 5, 2, 1),
        z_padding=(7, 5, 2, 1),
    ):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayerBN(
                        inChannels,
                        growth_Channels,
                        z_dilation=z_dilation[i % len(z_dilation)],
                        z_padding=z_padding[i % len(z_dilation)],
                    )
                )
                inChannels += growth_Channels
        layers.append(InPlaceABNSync(inChannels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.downblock1(x)
        x2 = self.denseblock1(x1)
        x3 = self.downblock2(x2)
        x4 = self.denseblock2(x3)
        x5 = self.downblock3(x4)
        x6 = self.denseblock3(x5)

        x7 = self.upblock3(x6)
        x8 = self.upblock2(torch.cat((x4, x7), dim=1))
        x9 = self.upblock1(torch.cat((x2, x8), dim=1))
        out = self.FCN_out(x9)

        return out


class Dense_Vnet_5(nn.Module):
    def __init__(self):
        super(Dense_Vnet_5, self).__init__()
        self.downblock1 = DownBlockBN(1, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock1 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=10,
            z_dilation=(7, 5, 2, 1),
            z_padding=(7, 5, 2, 1),
        )

        self.downblock2 = DownBlockBN(320, 32, kernel_size=4, stride=2, padding=1)
        self.denseblock2 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=10,
            z_dilation=(5, 2, 1, 1),
            z_padding=(5, 2, 1, 1),
        )

        self.downblock3 = DownBlockBN(
            320, 32, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.denseblock3 = self._make_denseBN(
            inChannels=32,
            growth_Channels=32,
            nDenseLayers=10,
            z_dilation=(2, 1, 1, 1),
            z_padding=(2, 1, 1, 1),
        )

        self.upblock1 = UpBlockBN(320 + 128, 64, kernel_size=4, stride=2, padding=1)
        self.upblock2 = UpBlockBN(320 + 128, 128, kernel_size=4, stride=2, padding=1)
        self.upblock3 = UpBlockBN(
            320, 128, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)
        )

        # self.FCN_out = PreConvBN3d(64, 3, kernel_size=3, padding=1)
        self.FCN_out = PreConvBN3d(64, 4, kernel_size=3, padding=1)

        print("finish building")

    def _make_denseBN(
        self,
        inChannels,
        growth_Channels,
        nDenseLayers,
        z_dilation=(7, 5, 2, 1),
        z_padding=(7, 5, 2, 1),
    ):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayerBN(
                        inChannels,
                        growth_Channels,
                        z_dilation=z_dilation[i % len(z_dilation)],
                        z_padding=z_padding[i % len(z_dilation)],
                    )
                )
                inChannels += growth_Channels
        layers.append(InPlaceABNSync(inChannels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.downblock1(x)
        x2 = self.denseblock1(x1)
        x3 = self.downblock2(x2)
        x4 = self.denseblock2(x3)
        x5 = self.downblock3(x4)
        x6 = self.denseblock3(x5)

        x7 = self.upblock3(x6)
        x8 = self.upblock2(torch.cat((x4, x7), dim=1))
        x9 = self.upblock1(torch.cat((x2, x8), dim=1))
        out = self.FCN_out(x9)

        # return torch.softmax(out, dim=1)
        return out


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


class UpBlockBN(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(UpBlockBN, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm3d(out_planes)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SingleLayerBN(nn.Module):
    def __init__(self, inChannels, growth_Channels, z_dilation=1, z_padding=1):
        super(SingleLayerBN, self).__init__()
        self.bn = InPlaceABNSync(inChannels)
        self.conv = nn.Conv3d(
            inChannels,
            growth_Channels,
            kernel_size=3,
            padding=(1, 1, z_padding),
            dilation=(1, 1, z_dilation),
        )

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = torch.cat((x, out), 1)
        return out


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


if __name__ == "__main__":
    model = Dense_Vnet_5().cuda()
    print(model(torch.randn([1, 1, 640, 96, 96]).cuda()))
    print("Test passed")
