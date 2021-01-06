import torch
from torch import nn
import copy
from .bp_inplace_abn import InPlaceABN, InPlaceABNSync
from .base_nets.layer import Bottleneck_inplace


class single_encoder(nn.Module):
    def __init__(
        self, n_inp=1, n_out=1, feats=(32, 32, 64, 64, 128, 128, 128), sync=False,
    ):

        super(single_encoder, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.n_inp = n_inp
        if sync:
            abnblock = InPlaceABNSync
        else:
            abnblock = InPlaceABN

        self.in_layer = nn.Conv3d(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = abnblock(feats[0])

        self.lconvlayer1 = nn.Conv3d(
            feats[0], feats[1], kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.lconvlayer1_bn = abnblock(feats[1])

        self.lconvlayer2 = nn.Conv3d(
            feats[1], feats[2], kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.lconvlayer2_bn = abnblock(feats[2])

        self.lconvlayer3 = nn.Conv3d(
            feats[2], feats[3], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer3_bn = abnblock(feats[3])

        self.lconvlayer4 = nn.Conv3d(
            feats[3], feats[4], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer4_bn = abnblock(feats[4])

        self.lconvlayer5 = nn.Conv3d(
            feats[4], feats[5], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer5_bn = abnblock(feats[5])

        self.lconvlayer6 = nn.Conv3d(
            feats[5], feats[6], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer6_bn = abnblock(feats[6])

    def forward(self, x):
        xl0 = self.in_layer_bn(self.in_layer(x))
        xl1 = self.lconvlayer1_bn(self.lconvlayer1(xl0))
        xl2 = self.lconvlayer2_bn(self.lconvlayer2(xl1))
        xl3 = self.lconvlayer3_bn(self.lconvlayer3(xl2))
        xl4 = self.lconvlayer4_bn(self.lconvlayer4(xl3))
        xl5 = self.lconvlayer5_bn(self.lconvlayer5(xl4))
        xl6 = self.lconvlayer6_bn(self.lconvlayer6(xl5))

        return xl0, xl1, xl2, xl3, xl4, xl5, xl6


class unet_shared_encoders(nn.Module):
    def __init__(
        self,
        n_inp=1,
        n_out=1,
        feats=(32, 32, 64, 64, 128, 128, 128),
        sync=False,
        n_encoders=2,
    ):
        super().__init__()

        self.previous_encoder = 0

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.n_inp = n_inp
        if sync:
            abnblock = InPlaceABNSync
        else:
            abnblock = InPlaceABN

        self.in_layer = nn.Conv3d(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = abnblock(feats[0])

        self.lconvlayer1 = nn.Conv3d(
            feats[0], feats[1], kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.lconvlayer1_bn = abnblock(feats[1])

        self.lconvlayer2 = nn.Conv3d(
            feats[1], feats[2], kernel_size=3, stride=(1, 2, 2), padding=1
        )
        self.lconvlayer2_bn = abnblock(feats[2])

        self.lconvlayer3 = nn.Conv3d(
            feats[2], feats[3], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer3_bn = abnblock(feats[3])

        self.lconvlayer4 = nn.Conv3d(
            feats[3], feats[4], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer4_bn = abnblock(feats[4])

        self.lconvlayer5 = nn.Conv3d(
            feats[4], feats[5], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer5_bn = abnblock(feats[5])

        self.lconvlayer6 = nn.Conv3d(
            feats[5], feats[6], kernel_size=4, stride=2, padding=1
        )
        self.lconvlayer6_bn = abnblock(feats[6])

        self.rconvlayer7 = nn.Conv3d(feats[6], feats[6], kernel_size=3, padding=1)
        self.rconvlayer7_bn = abnblock(feats[6])

        self.rconvTlayer6 = nn.Conv3d(feats[6], feats[5], kernel_size=3, padding=1)
        self.rconvTlayer6_bn = abnblock(feats[5])
        self.rconvlayer6 = nn.Conv3d(feats[5], feats[5], kernel_size=3, padding=1)
        self.rconvlayer6_bn = abnblock(feats[5])

        self.rconvTlayer5 = nn.Conv3d(feats[5], feats[4], kernel_size=3, padding=1)
        self.rconvTlayer5_bn = abnblock(feats[4])
        self.rconvlayer5 = nn.Conv3d(feats[4], feats[4], kernel_size=3, padding=1)
        self.rconvlayer5_bn = abnblock(feats[4])

        self.rconvTlayer4 = nn.Conv3d(feats[4], feats[3], kernel_size=3, padding=1)
        self.rconvTlayer4_bn = abnblock(feats[3])
        self.rconvlayer4 = nn.Conv3d(feats[3], feats[3], kernel_size=3, padding=1)
        self.rconvlayer4_bn = abnblock(feats[3])

        self.rconvTlayer3 = nn.Conv3d(feats[3], feats[2], kernel_size=3, padding=1)
        self.rconvTlayer3_bn = abnblock(feats[2])
        self.rconvlayer3 = nn.Conv3d(feats[2], feats[2], kernel_size=3, padding=1)
        self.rconvlayer3_bn = abnblock(feats[2])

        self.rconvTlayer2 = nn.Conv3d(feats[2], feats[1], kernel_size=3, padding=1)
        self.rconvTlayer2_bn = abnblock(feats[1])
        self.rconvlayer2 = nn.Conv3d(feats[1], feats[1], kernel_size=3, padding=1)
        self.rconvlayer2_bn = abnblock(feats[1])

        self.rconvTlayer1 = nn.Conv3d(feats[1], feats[0], kernel_size=3, padding=1)
        self.rconvTlayer1_bn = abnblock(feats[0])
        self.rconvlayer1 = nn.Conv3d(feats[0], feats[0], kernel_size=3, padding=1)
        self.rconvlayer1_bn = abnblock(feats[0])

        self.out_layer = nn.Conv3d(feats[0], n_out, kernel_size=1, stride=1)

    def forward(self, x, *args):

        # self.encoder_state_dict[self.previous_encoder] = self.encoder.state_dict()
        # sequence_index = args[0][0][0]
        # self.encoder.load_state_dict(self.encoder_state_dict[sequence_index])

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
            self.rconvTlayer3(self.upsample(self.rconvlayer4_bn(self.rconvlayer4(xr3))))
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


class unet_shared_encoders_medium(unet_shared_encoders):
    def __init__(self, sync=True):
        super().__init__(
            n_inp=1,
            n_out=1,
            feats=[8, 32, 48, 64, 128, 256, 512],
            sync=sync,
            n_encoders=6,
        )
