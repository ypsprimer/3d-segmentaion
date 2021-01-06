import torch
from torch import nn
from torch.nn import functional as F
from inplace_abn import InPlaceABN, InPlaceABNSync
from .layer import Bottleneck_inplace


class generator(nn.Module):
    def __init__(
        self,
        n_inp=1,
        n_out=1,
        feats=(32, 32, 64, 64, 128, 128, 128),
        sync=False,
        conv2d=nn.Conv2d,
        conv3d=nn.Conv3d,
    ):
        super(generator, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(2, 1, 1), mode="nearest")
        self.n_inp = n_inp
        if sync:
            abnblock = InPlaceABNSync
        else:
            abnblock = InPlaceABN

        self.in_layer = conv3d(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = abnblock(feats[0])

        self.lconvlayer1 = conv3d(
            feats[0],
            feats[1],
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        )  # 32,160
        self.lconvlayer1_bn = abnblock(feats[1])
        self.res1b = Bottleneck_inplace(feats[1], int(feats[1] / 4), abn=abnblock)

        self.lconvlayer2 = conv3d(
            feats[1],
            feats[2],
            kernel_size=(4, 4, 4),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
        )  # 32,80
        self.lconvlayer2_bn = abnblock(feats[2])
        self.res2b = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)
        self.res2c = Bottleneck_inplace(feats[2], int(feats[2] / 4), abn=abnblock)

        self.lconvlayer3 = conv3d(
            feats[2],
            feats[3],
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # 16,40
        self.lconvlayer3_bn = abnblock(feats[3])
        self.res3b = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)
        self.res3c = Bottleneck_inplace(feats[3], int(feats[3] / 4), abn=abnblock)

        self.lconvlayer4 = conv3d(
            feats[3],
            feats[4],
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # 8,20
        self.lconvlayer4_bn = abnblock(feats[4])
        self.res4b = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)
        self.res4c = Bottleneck_inplace(feats[4], int(feats[4] / 4), abn=abnblock)

        self.lconvlayer5 = conv3d(
            feats[4],
            feats[5],
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # 4,10
        self.lconvlayer5_bn = abnblock(feats[5])
        self.res5b = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)
        self.res5c = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)

        self.lconvlayer6 = conv3d(
            feats[5],
            feats[6],
            kernel_size=(4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )  # 4,10
        self.lconvlayer6_bn = abnblock(feats[6])
        self.res6b = Bottleneck_inplace(feats[6], int(feats[6] / 4), abn=abnblock)
        self.res6c = Bottleneck_inplace(feats[6], int(feats[6] / 4), abn=abnblock)

        self.rconvlayer7 = Bottleneck_inplace(feats[6], int(feats[6] / 4), abn=abnblock)

        self.rconvTlayer6 = conv3d(feats[6], feats[5], kernel_size=3, padding=1)
        self.rconvTlayer6_bn = abnblock(feats[5])
        self.rconvlayer6 = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)

        self.res6 = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)
        self.res6a = Bottleneck_inplace(feats[5], int(feats[5] / 4), abn=abnblock)

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

    def forward(self, x, _):

        xl0 = self.in_layer_bn(self.in_layer(x))
        xl1 = self.lconvlayer1_bn(self.lconvlayer1(xl0))
        xl1 = self.res1b(xl1)
        xl2 = self.lconvlayer2_bn(self.lconvlayer2(xl1))
        xl2 = self.res2b(xl2)
        xl2 = self.res2c(xl2)
        xl3 = self.lconvlayer3_bn(self.lconvlayer3(xl2))
        xl3 = self.res3b(xl3)
        xl3 = self.res3c(xl3)
        xl4 = self.lconvlayer4_bn(self.lconvlayer4(xl3))
        xl4 = self.res4b(xl4)
        xl4 = self.res4c(xl4)
        xl5 = self.lconvlayer5_bn(self.lconvlayer5(xl4))
        xl5 = self.res5b(xl5)
        xl5 = self.res5c(xl5)
        xl6 = self.lconvlayer6_bn(self.lconvlayer6(xl5))
        xl6 = self.res6b(xl6)
        xl6 = self.res6c(xl6)

        xr6 = self.rconvTlayer6_bn(
            self.rconvTlayer6(self.upsamplez((self.rconvlayer7(xl6))))
        )
        xr6 = self.res6(xr6)
        xr6 = self.res6a(xr6)

        xr5 = torch.add(xr6, xl5)
        xr5 = self.rconvTlayer5_bn(
            self.rconvTlayer5(self.upsamplez((self.rconvlayer6(xr5))))
        )
        xr5 = self.res5(xr5)
        xr5 = self.res5a(xr5)

        xr4 = torch.add(xr5, xl4)
        xr4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsamplez((self.rconvlayer5(xr4))))
        )
        xr4 = self.res4(xr4)
        xr4 = self.res4a(xr4)

        xr3 = torch.add(xr4, xl3)
        xr3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(self.upsamplez((self.rconvlayer4(xr3))))
        )
        xr3 = self.res3(xr3)
        xr3 = self.res3a(xr3)

        xr2 = torch.add(xr3, xl2)
        xr2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(self.upsample((self.rconvlayer3(xr2))))
        )
        xr2 = self.res2(xr2)
        xr2 = self.res2a(xr2)

        xr1 = torch.add(xr2, xl1)
        xr1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(self.upsample((self.rconvlayer2(xr1))))
        )
        # xr12 = checkpoint(self.res1, xr1)
        xr12 = self.res1(xr1)
        xr12 = self.res1a(xr12)

        xr0 = torch.add(xr12, xl0)
        xr0 = self.rconvlayer1(xr0)

        out_layer = self.out_layer(xr0)
        return torch.softmax(out_layer, dim=1)


class discriminator(nn.Module):
    def __init__(self, sync=False):
        super(discriminator, self).__init__()

        if sync:
            abnblock = InPlaceABNSync
        else:
            abnblock = InPlaceABN

        self.conv1 = nn.Conv3d(
            in_channels=5, out_channels=4, kernel_size=4, padding=1, stride=2
        )
        self.conv2 = nn.Conv3d(
            in_channels=4, out_channels=8, kernel_size=4, padding=1, stride=2
        )
        self.conv3 = nn.Conv3d(
            in_channels=8, out_channels=1, kernel_size=(3, 6, 6), padding=(1, 0, 0), stride=1
        )
        self.bn1 = abnblock(4)
        self.bn2 = abnblock(8)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:

        x = F.interpolate(torch.cat([input, label], dim=1), scale_factor=0.25)
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = torch.sigmoid(x)

        return x


class GAN_model(nn.Module):
    def __init__(self, sync=True):
        super(GAN_model, self).__init__()
        self.generator = generator(
            n_inp=1, n_out=4, feats=[16, 32, 64, 64, 128, 256, 512], sync=sync
        )
        self.discriminator = discriminator(sync=True)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> tuple:

        y = (
            torch.zeros([1] + ([int(label.max().long()) + 1]) + (list(input.shape[-3:])))
            .to(dtype=input.dtype, device=input.device)
            .scatter_(1, label.long(), value=1)
        )
        pred = self.generator(input, y)
        dis_pred = self.discriminator(input, pred.detach())
        dis_gt = self.discriminator(input, y)

        return tuple([pred, dis_pred, dis_gt])


class unet_c1_e(generator):
    def __init__(self, sync=False):
        super(unet_c1_e, self).__init__(
            n_inp=1, feats=[16, 32, 64, 64, 128, 128, 128], sync=sync
        )


class unet_c1_e_sync(generator):
    def __init__(self, sync=True):
        super(unet_c1_e_sync, self).__init__(
            n_inp=1, feats=[16, 32, 64, 64, 128, 128, 128], sync=sync
        )


class unet_c1_f_sync(generator):
    def __init__(self, sync=True):
        super(unet_c1_f_sync, self).__init__(
            n_inp=1, n_out=4, feats=[16, 32, 64, 64, 128, 256, 512], sync=sync
        )
