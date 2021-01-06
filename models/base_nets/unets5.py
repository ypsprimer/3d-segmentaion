import torch
from torch import nn

from models.bp_inplace_abn import InPlaceABN, InPlaceABNSync
from models.base_nets.layer import Bottleneck_inplace


class unet_type5(nn.Module):
    def __init__(
        self,
        n_inp=1,
        n_out=1,
        feats=[32, 32, 64, 64, 128, 128, 128],
        sync=False,
        conv2d=nn.Conv2d,
        conv3d=nn.Conv3d,
    ):
        super(unet_type5, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(1.0, 2.0, 2.0), mode="nearest")
        self.n_inp = n_inp
        if sync:
            abnblock = InPlaceABNSync
        else:
            abnblock = InPlaceABN

        self.in_layer = conv3d(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = abnblock(feats[0])

        self.res0b = Bottleneck_inplace(feats[0], int(feats[0] / 4), abn=abnblock)

        self.lconvlayer1 = conv3d(
            feats[0],
            feats[1],
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )  # 32,160
        self.lconvlayer1_bn = abnblock(feats[1])
        self.res1b = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)
        self.res1c = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)

        self.lconvlayer2 = conv3d(
            feats[1],
            feats[2],
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )  # 32,80
        self.lconvlayer2_bn = abnblock(feats[2])
        self.res2b = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)
        self.res2c = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)

        self.lconvlayer3 = conv3d(
            feats[2], feats[3], kernel_size=4, stride=2, padding=1
        )  # 16,40
        self.lconvlayer3_bn = abnblock(feats[3])
        self.res3b = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)
        self.res3c = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)

        self.lconvlayer4 = conv3d(
            feats[3], feats[4], kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = abnblock(feats[4])
        self.res4b = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)
        self.res4c = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)

        self.lconvlayer5 = conv3d(
            feats[4], feats[5], kernel_size=4, stride=2, padding=1
        )  # 4,10
        self.lconvlayer5_bn = abnblock(feats[5])
        self.res5b = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)
        self.res5c = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)

        self.lconvlayer6 = conv3d(
            feats[5], feats[6], kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = abnblock(feats[6])

        self.rconvlayer7 = conv3d(feats[6], feats[6], kernel_size=3, padding=1)
        self.rconvlayer7_bn = abnblock(feats[6])

        self.rconvTlayer6 = conv3d(feats[6], feats[5], kernel_size=3, padding=1)
        self.rconvTlayer6_bn = abnblock(feats[5])
        self.rconvlayer6 = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)

        self.rconvTlayer5 = conv3d(feats[5], feats[4], kernel_size=3, padding=1)
        self.rconvTlayer5_bn = abnblock(feats[4])
        self.rconvlayer5 = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)

        self.res5 = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)
        self.res5a = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)

        self.rconvTlayer4 = conv3d(feats[4], feats[3], kernel_size=3, padding=1)
        self.rconvTlayer4_bn = abnblock(feats[3])
        self.rconvlayer4 = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)

        self.res4 = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)
        self.res4a = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)

        self.rconvTlayer3 = conv3d(feats[3], feats[2], kernel_size=3, padding=1)
        self.rconvTlayer3_bn = abnblock(feats[2])
        self.rconvlayer3 = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)

        self.res3 = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)
        self.res3a = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = conv3d(feats[2], feats[1], kernel_size=3, padding=1)
        self.rconvTlayer2_bn = abnblock(feats[1])
        self.rconvlayer2 = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)

        self.res2 = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)
        self.res2a = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)

        self.rconvTlayer1 = conv3d(feats[1], feats[0], kernel_size=3, padding=1)
        self.rconvTlayer1_bn = abnblock(feats[0])
        self.rconvlayer1 = Bottleneck_inplace(feats[0], int(feats[0] / 4), abn=abnblock)
        # self.res1 = ResBlock2(feats[0], feats[0])
        self.res1 = Bottleneck_inplace(feats[0], int(feats[0] / 4), abn=abnblock)
        self.res1a = Bottleneck_inplace(feats[0], int(feats[0] / 4), abn=abnblock)


        self.out_layer = conv3d(feats[0], n_out, kernel_size=1, stride=1)

    def forward(self, x, deploy=0):
        # if deploy:
        #     return self.deploy(x)
        #        xlact = self.act1(x)
        #        xl0 = self.relu(self.in_layer_bn(self.in_layer(xlact)))
        x0 = self.in_layer_bn(self.in_layer(x))
        x0 = self.res0b(x0)
        x1 = self.lconvlayer1_bn(self.lconvlayer1(x0))
        x1 = self.res1b(x1)
        x1 = self.res1c(x1)
        x2 = self.lconvlayer2_bn(self.lconvlayer2(x1))
        x2 = self.res2b(x2)
        x2 = self.res2c(x2)
        x3 = self.lconvlayer3_bn(self.lconvlayer3(x2))
        x3 = self.res3b(x3)
        x3 = self.res3c(x3)
        x4 = self.lconvlayer4_bn(self.lconvlayer4(x3))
        x4 = self.res4b(x4)
        x4 = self.res4c(x4)
        x5 = self.lconvlayer5_bn(self.lconvlayer5(x4))
        x5 = self.res5b(x5)
        x5 = self.res5c(x5)
        x6 = self.lconvlayer6_bn(self.lconvlayer6(x5))

        x6 = self.rconvTlayer6_bn(
            self.rconvTlayer6(self.upsample(self.rconvlayer7_bn(self.rconvlayer7(x6))))
        )

        x5 = torch.add(x6, x5)
        x5 = self.rconvTlayer5_bn(
            self.rconvTlayer5(self.upsample((self.rconvlayer6(x5))))
        )
        # xr5 = checkpoint(self.res5, xr5)
        x5 = self.res5(x5)
        x5 = self.res5a(x5)

        x4 = torch.add(x5, x4)
        x4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsample((self.rconvlayer5(x4))))
        )
        # xr4 = checkpoint(self.res4, xr4)
        x4 = self.res4(x4)
        x4 = self.res4a(x4)

        x3 = torch.add(x4, x3)
        x3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(self.upsample((self.rconvlayer4(x3))))
        )
        # xr3 = checkpoint(self.res3, xr3)
        x3 = self.res3(x3)
        x3 = self.res3a(x3)

        x2 = torch.add(x3, x2)
        x2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(self.upsamplez((self.rconvlayer3(x2))))
        )
        # xr2 = checkpoint(self.res2, xr2)
        x2 = self.res2(x2)
        x2 = self.res2a(x2)

        x1 = torch.add(x2, x1)
        x1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(self.upsamplez((self.rconvlayer2(x1))))
        )
        # xr12 = checkpoint(self.res1, xr1)
        x1 = self.res1(x1)
        x1 = self.res1a(x1)

        x0 = torch.add(x0, x1)
        x0 = self.rconvlayer1(x0)

        x0 = self.out_layer(x0)

        return torch.softmax(x0, dim=1)

    def deploy(self, x):

        x0 = self.in_layer_bn(self.in_layer(x))
        x0 = self.res0b(x0)
        x1 = self.lconvlayer1_bn(self.lconvlayer1(x0))
        x1 = self.res1b(x1)
        x1 = self.res1c(x1)
        x2 = self.lconvlayer2_bn(self.lconvlayer2(x1))
        x2 = self.res2b(x2)
        x2 = self.res2c(x2)
        x3 = self.lconvlayer3_bn(self.lconvlayer3(x2))
        x3 = self.res3b(x3)
        x3 = self.res3c(x3)
        x4 = self.lconvlayer4_bn(self.lconvlayer4(x3))
        x4 = self.res4b(x4)
        x4 = self.res4c(x4)
        x5 = self.lconvlayer5_bn(self.lconvlayer5(x4))
        x5 = self.res5b(x5)
        x5 = self.res5c(x5)
        x6 = self.lconvlayer6_bn(self.lconvlayer6(x5))

        x6 = self.rconvTlayer6_bn(
            self.rconvTlayer6(self.upsample(self.rconvlayer7_bn(self.rconvlayer7(x6))))
        )

        x5.add_(x6)
        del x6
        x5 = self.rconvTlayer5_bn(
            self.rconvTlayer5(self.upsample((self.rconvlayer6(x5))))
        )
        # xr5 = checkpoint(self.res5, xr5)
        x5 = self.res5(x5)
        x5 = self.res5a(x5)

        x4.add_(x5)
        del x5
        x4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsample((self.rconvlayer5(x4))))
        )
        # xr4 = checkpoint(self.res4, xr4)
        x4 = self.res4(x4)
        x4 = self.res4a(x4)

        x3.add_(x4)
        del x4
        x3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(self.upsample((self.rconvlayer4(x3))))
        )
        # xr3 = checkpoint(self.res3, xr3)
        x3 = self.res3(x3)
        x3 = self.res3a(x3)

        x2.add_(x3)
        del x3
        x2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(self.upsamplez((self.rconvlayer3(x2))))
        )
        x2 = self.res2(x2)
        x2 = self.res2a(x2)

        x1.add_(x2)
        del x2
        x1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(self.upsamplez((self.rconvlayer2(x1))))
        )
        # xr12 = checkpoint(self.res1, xr1)
        x1 = self.res1(x1)
        x1 = self.res1a(x1)

        x0.add_(x1)
        del x1
        x0 = self.rconvlayer1(x0)

        x0 = self.out_layer(x0)
        return torch.sigmoid(x0)
        # xr2 = checkpoint(self.res2, xr2)


class unet_c1_d(unet_type5):
    def __init__(self, sync=False):
        super().__init__(n_out=1, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c1_d_sync(unet_type5):
    def __init__(self, sync=True):
        super().__init__(n_out=1, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c2_d(unet_type5):
    def __init__(self, sync=False):
        super().__init__(n_out=2, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c2_d_sync(unet_type5):
    def __init__(self, sync=True):
        super().__init__(n_out=2, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c3_d(unet_type5):
    def __init__(self, sync=False):
        super().__init__(n_out=3, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c3_d_sync(unet_type5):
    def __init__(self, sync=True):
        super().__init__(n_out=3, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync)


class unet_c3_e_sync(unet_type5):
    def __init__(self, sync=True):
        super().__init__(n_out=3, feats=[64, 128, 256, 512, 512, 512, 512], sync=sync)
