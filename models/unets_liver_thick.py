import torch
from torch import nn

from .inplace_abn import ABN, InPlaceABN, InPlaceABNSync


class unet_type_liver_thick(nn.Module):
    def __init__(
        self,
        n_inp=1,
        feats=(32, 32, 64, 64, 128, 128, 128),
        abn=1,
        conv2d=nn.Conv2d,
        conv3d=nn.Conv3d,
    ):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.n_inp = n_inp
        if abn == 0:
            abnblock = ABN
        elif abn == 1:
            abnblock = InPlaceABN
        else:
            abnblock = InPlaceABNSync

        self.in_layer = conv3d(
            n_inp, feats[0], kernel_size=(1, 5, 5), padding=(0, 2, 2)
        )  # 32,320
        self.in_layer_bn = abnblock(feats[0])

        self.lconvlayer1 = conv3d(
            feats[0],
            feats[1],
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )  # 32,160
        self.lconvlayer1_bn = abnblock(feats[1])

        self.lconvlayer2 = conv3d(
            feats[1],
            feats[2],
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )  # 32,80
        self.lconvlayer2_bn = abnblock(feats[2])

        self.lconvlayer3 = conv3d(
            feats[2],
            feats[3],
            kernel_size=(1, 4, 4),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )  # 16,40
        self.lconvlayer3_bn = abnblock(feats[3])

        self.lconvlayer4 = conv3d(
            feats[3], feats[4], kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = abnblock(feats[4])

        self.lconvlayer5 = conv3d(
            feats[4], feats[5], kernel_size=4, stride=2, padding=1
        )  # 4,10
        self.lconvlayer5_bn = abnblock(feats[5])

        self.lconvlayer6 = conv3d(
            feats[5], feats[6], kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = abnblock(feats[6])

        # self.lconvlayer7 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7_bn = InPlaceABN(256)
        #
        # self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = conv3d(feats[6], feats[6], kernel_size=3, padding=1)
        self.rconvlayer7_bn = abnblock(feats[6])

        self.rconvTlayer6 = conv3d(feats[6], feats[5], kernel_size=3, padding=1)
        self.rconvTlayer6_bn = abnblock(feats[5])
        self.rconvlayer6 = conv3d(feats[5], feats[5], kernel_size=3, padding=1)
        self.rconvlayer6_bn = abnblock(feats[5])

        self.rconvTlayer5 = conv3d(feats[5], feats[4], kernel_size=3, padding=1)
        self.rconvTlayer5_bn = abnblock(feats[4])
        self.rconvlayer5 = conv3d(feats[4], feats[4], kernel_size=3, padding=1)
        self.rconvlayer5_bn = abnblock(feats[4])

        self.rconvTlayer4 = conv3d(feats[4], feats[3], kernel_size=3, padding=1)
        self.rconvTlayer4_bn = abnblock(feats[3])
        self.rconvlayer4 = conv3d(feats[3], feats[3], kernel_size=3, padding=1)
        self.rconvlayer4_bn = abnblock(feats[3])

        self.rconvTlayer3 = conv3d(
            feats[3], feats[2], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvTlayer3_bn = abnblock(feats[2])
        self.rconvlayer3 = conv3d(
            feats[2], feats[2], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvlayer3_bn = abnblock(feats[2])

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = conv3d(
            feats[2], feats[1], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvTlayer2_bn = abnblock(feats[1])
        self.rconvlayer2 = conv3d(
            feats[1], feats[1], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvlayer2_bn = abnblock(feats[1])

        self.rconvTlayer1 = conv3d(
            feats[1], feats[0], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvTlayer1_bn = abnblock(feats[0])
        self.rconvlayer1 = conv3d(
            feats[0], feats[0], kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        self.rconvlayer1_bn = abnblock(feats[0])

        self.out_layer = conv3d(feats[0], 1, kernel_size=1, stride=1)

    def forward(self, x):

        #        xlact = self.act1(x)
        #        xl0 = self.relu(self.in_layer_bn(self.in_layer(xlact)))
        xl0 = self.in_layer_bn(self.in_layer(x))
        xl1 = self.lconvlayer1_bn(self.lconvlayer1(xl0))
        xl2 = self.lconvlayer2_bn(self.lconvlayer2(xl1))
        xl3 = self.lconvlayer3_bn(self.lconvlayer3(xl2))
        xl4 = self.lconvlayer4_bn(self.lconvlayer4(xl3))
        xl5 = self.lconvlayer5_bn(self.lconvlayer5(xl4))
        xl6 = self.lconvlayer6_bn(self.lconvlayer6(xl5))

        xr6 = xl6
        xr61 = self.rconvTlayer6_bn(
            self.rconvTlayer6(self.upsample(self.rconvlayer7_bn(self.rconvlayer7(xr6))))
        )

        xr51 = torch.add(xr61, xl5)
        xr5 = self.rconvTlayer5_bn(
            self.rconvTlayer5(
                self.upsample(self.rconvlayer6_bn(self.rconvlayer6(xr51)))
            )
        )
        xr4 = torch.add(xr5, xl4)

        xr4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsample(self.rconvlayer5_bn(self.rconvlayer5(xr4))))
        )

        xr3 = torch.add(xr4, xl3)
        xr3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(
                self.upsamplez(self.rconvlayer4_bn(self.rconvlayer4(xr3)))
            )
        )

        xr2 = torch.add(xr3, xl2)
        xr2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(
                self.upsamplez(self.rconvlayer3_bn(self.rconvlayer3(xr2)))
            )
        )

        xr1 = torch.add(xr2, xl1)
        xr1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(
                self.upsamplez(self.rconvlayer2_bn(self.rconvlayer2(xr1)))
            )
        )

        xr0 = torch.add(xr1, xl0)

        xr0 = self.rconvlayer1_bn(self.rconvlayer1(xr0))
        out_layer = self.out_layer(xr0)

        return torch.sigmoid(out_layer)

    def deploy(self, x0, extreme=False):
        x0 = self.in_layer_bn(self.in_layer(x0))
        x1 = self.lconvlayer1_bn(self.lconvlayer1(x0))
        x2 = self.lconvlayer2_bn(self.lconvlayer2(x1))
        x3 = self.lconvlayer3_bn(self.lconvlayer3(x2))
        x4 = self.lconvlayer4_bn(self.lconvlayer4(x3))
        x5 = self.lconvlayer5_bn(self.lconvlayer5(x4))
        x6 = self.lconvlayer6_bn(self.lconvlayer6(x5))

        x6 = self.rconvTlayer6_bn(
            self.rconvTlayer6(self.upsample(self.rconvlayer7_bn(self.rconvlayer7(x6))))
        )

        x5.add_(x6)
        if extreme:
            del x6
        x5 = self.rconvTlayer5_bn(
            self.rconvTlayer5(self.upsample(self.rconvlayer6_bn(self.rconvlayer6(x5))))
        )

        x4.add_(x5)
        if extreme:
            del x5
        x4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsample(self.rconvlayer5_bn(self.rconvlayer5(x4))))
        )

        x3.add_(x4)
        if extreme:
            del x4
        x3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(self.upsample(self.rconvlayer4_bn(self.rconvlayer4(x3))))
        )

        x2.add_(x3)
        if extreme:
            del x3
        x2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(self.upsamplez(self.rconvlayer3_bn(self.rconvlayer3(x2))))
        )

        x1.add_(x2)
        if extreme:
            del x2
        x1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(self.upsamplez(self.rconvlayer2_bn(self.rconvlayer2(x1))))
        )

        x0.add_(x1)
        if extreme:
            del x1
        x0 = self.rconvlayer1_bn(self.rconvlayer1(x0))
        x0 = self.out_layer(x0)

        return x0


class unet_a_liver_thick(unet_type_liver_thick):
    def __init__(self, abn=1):
        super().__init__(n_inp=1, feats=[32, 32, 64, 64, 128, 128, 128], abn=abn)


class unet_a_liver_thick_sync(unet_type_liver_thick):
    def __init__(self, abn=2):
        super().__init__(n_inp=1, feats=[32, 32, 64, 64, 128, 128, 128], abn=abn)


class unet_a_liver_thick_sync_plus(unet_type_liver_thick):
    def __init__(self, abn=2):
        super().__init__(n_inp=1, feats=[32, 64, 128, 128, 256, 512, 512], abn=abn)


