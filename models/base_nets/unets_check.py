import torch
from torch import nn
from inplace_abn import InPlaceABN, InPlaceABNSync
from base_nets.layer import Basic_block_inplace


class unet_type3(nn.Module):
    def __init__(
        self,
        n_inp=1,
        feats=[32, 32, 64, 64, 128, 128, 128],
        sync=False,
        conv2d=nn.Conv2d,
        conv3d=nn.Conv3d,
    ):
        super(unet_type3, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
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
            kernel_size=(5, 5, 5),
            stride=(1, 2, 2),
            padding=(2, 2, 2),
        )  # 32,160
        self.lconvlayer1_bn = abnblock(feats[1])

        self.lconvlayer2 = conv3d(
            feats[1],
            feats[2],
            kernel_size=(5, 5, 5),
            stride=(1, 2, 2),
            padding=(2, 2, 2),
        )  # 32,80
        self.lconvlayer2_bn = abnblock(64)

        self.lconvlayer3 = conv3d(
            feats[2], feats[3], kernel_size=4, stride=2, padding=1
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

        self.res5 = Basic_block_inplace(feats[4], feats[4], abn=abnblock)
        self.res5a = Basic_block_inplace(feats[4], feats[4], abn=abnblock)

        self.rconvTlayer4 = conv3d(feats[4], feats[3], kernel_size=5, padding=2)
        self.rconvTlayer4_bn = abnblock(feats[3])
        self.rconvlayer4 = conv3d(feats[3], feats[3], kernel_size=5, padding=2)
        self.rconvlayer4_bn = abnblock(feats[3])

        self.res4 = Basic_block_inplace(feats[3], feats[3], abn=abnblock)
        self.res4a = Basic_block_inplace(feats[3], feats[3], abn=abnblock)

        self.rconvTlayer3 = conv3d(feats[3], feats[2], kernel_size=5, padding=2)
        self.rconvTlayer3_bn = abnblock(feats[2])
        self.rconvlayer3 = conv3d(feats[2], feats[2], kernel_size=5, padding=2)
        self.rconvlayer3_bn = abnblock(feats[2])

        self.res3 = Basic_block_inplace(feats[2], feats[2], abn=abnblock)
        self.res3a = Basic_block_inplace(feats[2], feats[2], abn=abnblock)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = conv3d(feats[2], feats[1], kernel_size=5, padding=2)
        self.rconvTlayer2_bn = abnblock(feats[1])
        self.rconvlayer2 = conv3d(feats[1], feats[1], kernel_size=5, padding=2)
        self.rconvlayer2_bn = abnblock(feats[1])

        self.res2 = Basic_block_inplace(feats[1], feats[1], abn=abnblock)
        self.res2a = Basic_block_inplace(feats[1], feats[1], abn=abnblock)

        self.rconvTlayer1 = conv3d(feats[1], feats[0], kernel_size=5, padding=2)
        self.rconvTlayer1_bn = abnblock(feats[0])
        self.rconvlayer1 = conv3d(feats[0], feats[0], kernel_size=5, padding=2)
        self.rconvlayer1_bn = abnblock(feats[0])
        # self.res1 = ResBlock2(feats[0], feats[0])
        self.res1 = Basic_block_inplace(feats[0], feats[0], abn=abnblock)
        self.res1a = Basic_block_inplace(feats[0], feats[0], abn=abnblock)

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
        # xr5 = checkpoint(self.res5, xr5)
        xr5 = self.res5(xr5)
        xr5 = self.res5a(xr5)

        xr4 = torch.add(xr5, xl4)
        xr4 = self.rconvTlayer4_bn(
            self.rconvTlayer4(self.upsample(self.rconvlayer5_bn(self.rconvlayer5(xr4))))
        )
        # xr4 = checkpoint(self.res4, xr4)
        xr4 = self.res4(xr4)
        xr4 = self.res4a(xr4)

        xr3 = torch.add(xr4, xl3)
        xr3 = self.rconvTlayer3_bn(
            self.rconvTlayer3(self.upsample(self.rconvlayer4_bn(self.rconvlayer4(xr3))))
        )
        # xr3 = checkpoint(self.res3, xr3)
        xr3 = self.res3(xr3)
        xr3 = self.res3a(xr3)

        xr2 = torch.add(xr3, xl2)
        xr2 = self.rconvTlayer2_bn(
            self.rconvTlayer2(
                self.upsamplez(self.rconvlayer3_bn(self.rconvlayer3(xr2)))
            )
        )
        # xr2 = checkpoint(self.res2, xr2)
        xr2 = self.res2(xr2)
        xr2 = self.res2a(xr2)

        xr1 = torch.add(xr2, xl1)
        xr1 = self.rconvTlayer1_bn(
            self.rconvTlayer1(
                self.upsamplez(self.rconvlayer2_bn(self.rconvlayer2(xr1)))
            )
        )
        # xr12 = checkpoint(self.res1, xr1)
        xr12 = self.res1(xr1)
        xr12 = self.res1a(xr12)

        xr0 = torch.add(xr12, xl0)
        xr0 = self.rconvlayer1_bn(self.rconvlayer1(xr0))

        out_layer = self.out_layer(xr0)
        return out_layer


class unet_c1_b(unet_type3):
    def __init__(self, sync=False):
        super(unet_c1_b, self).__init__(
            n_inp=1, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync
        )


class unet_c1_b_sync(unet_type3):
    def __init__(self, sync=True):
        super(unet_c1_b_sync, self).__init__(
            n_inp=1, feats=[32, 32, 64, 64, 128, 128, 128], sync=sync
        )


#
# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np
# from .inplace_abn import InPlaceABN, InPlaceABNSync
#
#
# class unet_type2(nn.Module):
#     def __init__(self, n_inp = 1, feats = [32, 32, 64, 64, 128, 128, 128, 128],
#                  forw_ks= [5, 5, 4, 4, 4, 4, 3],
#                  stride_list = [(1,2,2), (1,2,2), 2, 2, 2, 2, 1]):
#         super(unet_type2, self).__init__()
#
#         self.relu = nn.ReLU(inplace=True)
#         self.pixel_shuffle = nn.PixelShuffle(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
#         #        self.act1 = my_act()
#
#
#         self.in_layer = nn.Conv3d(n_inp, feats[0], kernel_size=5, padding=2)  # 32,320
#         self.in_layer_bn = InPlaceABNSync(feats[0])
#         def npad(x):
#             return int(np.ceil(x-1)/2)
#
#         for i in range(len(feats)-1):
#             setattr(self,'lconvlayer' + str(i+1), nn.Conv3d(feats[i], feats[i+1], kernel_size=forw_ks[i], stride=stride_list[i], padding=npad(forw_ks[i])))
#             setattr(self,'lconvlayer' + str(i+1)+'_bn', InPlaceABNSync(feats[i+1]))
#
#         self.rconvlayer7 = nn.Conv3d(feats[6], feats[6], kernel_size=3, padding=1)
#         self.rconvlayer7_bn = InPlaceABNSync(feats[6])
#
#         self.rconvTlayer6 = nn.Conv3d(feats[6], feats[5], kernel_size=3, padding=1)
#         self.rconvTlayer6_bn = InPlaceABNSync(feats[5])
#         self.rconvlayer6 = nn.Conv3d(feats[5], feats[5], kernel_size=3, padding=1)
#         self.rconvlayer6_bn = InPlaceABNSync(feats[5])
#
#         self.rconvTlayer5 = nn.Conv3d(feats[5], feats[4], kernel_size=3, padding=1)
#         self.rconvTlayer5_bn = InPlaceABNSync(feats[4])
#         self.rconvlayer5 = nn.Conv3d(feats[4], feats[4], kernel_size=3, padding=1)
#         self.rconvlayer5_bn = InPlaceABNSync(feats[4])
#
#         self.rconvTlayer4 = nn.Conv3d(feats[4], feats[3], kernel_size=5, padding=2)
#         self.rconvTlayer4_bn = InPlaceABNSync(feats[3])
#         self.rconvlayer4 = nn.Conv3d(feats[3], feats[3], kernel_size=5, padding=2)
#         self.rconvlayer4_bn = InPlaceABNSync(feats[3])
#
#         self.rconvTlayer3 = nn.Conv3d(feats[3], feats[2], kernel_size=5, padding=2)
#         self.rconvTlayer3_bn = InPlaceABNSync(feats[2])
#         self.rconvlayer3 = nn.Conv3d(feats[2], feats[2], kernel_size=5, padding=2)
#         self.rconvlayer3_bn = InPlaceABNSync(feats[2])
#
#         #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
#         self.rconvTlayer2 = nn.Conv3d(feats[2], feats[1], kernel_size=5, padding=2)
#         self.rconvTlayer2_bn = InPlaceABNSync(feats[1])
#         self.rconvlayer2 = nn.Conv3d(feats[1], feats[1], kernel_size=5, padding=2)
#         self.rconvlayer2_bn = InPlaceABNSync(feats[1])
#
#         self.rconvTlayer1 = nn.Conv3d(feats[1], feats[0], kernel_size=5, padding=2)
#         self.rconvTlayer1_bn = InPlaceABNSync(feats[0])
#         self.rconvlayer1 = nn.Conv3d(feats[0], feats[0], kernel_size=5, padding=2)
#         self.rconvlayer1_bn = InPlaceABNSync(feats[0])
#
#         self.out_layer = nn.Conv3d(feats[0], 1, kernel_size=1, stride=1)
#
#     def forward(self, x):
#         #        xlact = self.act1(x)
#         #        xl0 = self.relu(self.in_layer_bn(self.in_layer(xlact)))
#         xl0 = self.in_layer_bn(self.in_layer(x))
#         xl1 = self.lconvlayer1_bn(self.lconvlayer1(xl0))
#         xl2 = self.lconvlayer2_bn(self.lconvlayer2(xl1))
#         xl3 = self.lconvlayer3_bn(self.lconvlayer3(xl2))
#         xl4 = self.lconvlayer4_bn(self.lconvlayer4(xl3))
#         xl5 = self.lconvlayer5_bn(self.lconvlayer5(xl4))
#         xl6 = self.lconvlayer6_bn(self.lconvlayer6(xl5))
#
#         xr6 = xl6
#         xr61 = self.rconvTlayer6_bn(self.rconvTlayer6(self.upsample(
#             self.rconvlayer7_bn(self.rconvlayer7(xr6)))))
#
#         xr51 = torch.add(xr61, xl5)
#         xr5 = self.rconvTlayer5_bn(self.rconvTlayer5(self.upsample(
#             self.rconvlayer6_bn(self.rconvlayer6(xr51)))))
#         xr4 = torch.add(xr5, xl4)
#
#         xr4 = self.rconvTlayer4_bn(self.rconvTlayer4(self.upsample(
#                 self.rconvlayer5_bn(self.rconvlayer5(xr4)))))
#
#         xr3 = torch.add(xr4, xl3)
#         xr3 = self.rconvTlayer3_bn(self.rconvTlayer3(self.upsample(
#             self.rconvlayer4_bn(self.rconvlayer4(xr3)))))
#
#         xr2 = torch.add(xr3, xl2)
#         xr2 = self.rconvTlayer2_bn(self.rconvTlayer2(self.upsamplez(
#             self.rconvlayer3_bn(self.rconvlayer3(xr2)))))
#
#         xr1 = torch.add(xr2, xl1)
#         xr1 = self.rconvTlayer1_bn(self.rconvTlayer1(self.upsamplez(
#             self.rconvlayer2_bn(self.rconvlayer2(xr1)))))
#
#         xr0 = torch.add(xr1, xl0)
#
#         xr0 = self.rconvlayer1_bn(self.rconvlayer1(xr0))
#         out_layer = self.out_layer(xr0)
#
#         return out_layer
