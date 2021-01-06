import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from inplace_abn import InPlaceABN, InPlaceABNSync


class SKUNET100_DICE_BN_NEW_MOD4_EX(nn.Module):
    def __init__(self):
        super(SKUNET100_DICE_BN_NEW_MOD4_EX, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        #        self.act1 = my_act()

        self.in_layer = nn.Conv3d(1, 32, kernel_size=3, padding=1)  # 32,320
        self.in_layer_bn = nn.BatchNorm3d(32)

        self.lconvlayer1 = nn.Conv3d(
            32, 32, kernel_size=3, stride=(1, 2, 2), padding=1
        )  # 32,160

        self.lconvlayer2 = nn.Conv3d(
            32, 64, kernel_size=3, stride=(1, 2, 2), padding=1
        )  # 32,80
        self.lconvlayer2_bn = nn.BatchNorm3d(64)

        self.lconvlayer3 = nn.Conv3d(
            64, 64, kernel_size=3, stride=2, padding=1
        )  # 16,40

        self.lconvlayer4 = nn.Conv3d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = nn.BatchNorm3d(128)

        self.lconvlayer5 = nn.Conv3d(
            128, 128, kernel_size=3, stride=2, padding=1
        )  # 4,10

        self.lconvlayer6 = nn.Conv3d(
            128, 128, kernel_size=3, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = nn.BatchNorm3d(128)

        self.lconvlayer7 = nn.Conv3d(
            128, 256, kernel_size=3, stride=2, padding=1
        )  # 2, 5

        self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer7_bn = nn.BatchNorm3d(128)

        self.rconvTlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.rconvTlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5_bn = nn.BatchNorm3d(128)

        self.rconvTlayer4 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.rconvlayer4 = nn.Conv3d(64, 64, kernel_size=3, padding=1)

        self.rconvTlayer3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.rconvlayer3 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.rconvlayer3_bn = nn.BatchNorm3d(64)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.rconvlayer2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)

        self.rconvTlayer1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        #        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvlayer1 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.rconvlayer1_bn = nn.BatchNorm3d(32)

        self.out_layer = nn.Conv3d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        #        xlact = self.act1(x)
        #        xl0 = self.relu(self.in_layer_bn(self.in_layer(xlact)))
        xl0 = self.relu(self.in_layer_bn(self.in_layer(x)))
        xl1 = self.relu(self.lconvlayer1(xl0))
        xl2 = self.relu(self.lconvlayer2_bn(self.lconvlayer2(xl1)))
        xl3 = self.relu(self.lconvlayer3(xl2))
        xl4 = self.relu(self.lconvlayer4_bn(self.lconvlayer4(xl3)))
        xl5 = self.relu(self.lconvlayer5(xl4))
        xl6 = self.relu(self.lconvlayer6_bn(self.lconvlayer6(xl5)))

        xr6 = xl6
        xr61 = self.relu(
            self.rconvTlayer6(
                self.upsample(self.relu(self.rconvlayer7_bn(self.rconvlayer7(xr6))))
            )
        )

        xr51 = torch.add(xr61, xl5)
        xr5 = self.relu(
            self.rconvTlayer5(self.upsample(self.relu(self.rconvlayer6(xr51))))
        )
        xr4 = torch.add(xr5, xl4)
        xr4 = self.relu(
            self.rconvTlayer4(
                self.upsample(self.relu(self.rconvlayer5_bn(self.rconvlayer5(xr4))))
            )
        )

        xr3 = torch.add(xr4, xl3)
        xr3 = self.relu(
            self.rconvTlayer3(self.upsample(self.relu(self.rconvlayer4(xr3))))
        )

        xr2 = torch.add(xr3, xl2)
        xr2 = self.relu(
            self.rconvTlayer2(
                self.upsamplez(self.relu(self.rconvlayer3_bn(self.rconvlayer3(xr2))))
            )
        )

        xr1 = torch.add(xr2, xl1)
        xr1 = self.relu(
            self.rconvTlayer1(self.upsamplez(self.relu(self.rconvlayer2(xr1))))
        )

        xr0 = torch.add(xr1, xl0)

        xr0 = self.relu(self.rconvlayer1_bn(self.rconvlayer1(xr0)))
        out_layer = self.out_layer(xr0)

        return out_layer


class SKUNET100_DICE_BN_NEW_MOD4_EX2(nn.Module):
    def __init__(self):
        super(SKUNET100_DICE_BN_NEW_MOD4_EX2, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        #        self.act1 = my_act()

        self.in_layer = nn.Conv3d(1, 32, kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = nn.BatchNorm3d(32)

        self.lconvlayer1 = nn.Conv3d(
            32, 32, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,160

        self.lconvlayer2 = nn.Conv3d(
            32, 64, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,80
        self.lconvlayer2_bn = nn.BatchNorm3d(64)

        self.lconvlayer3 = nn.Conv3d(
            64, 64, kernel_size=4, stride=2, padding=1
        )  # 16,40

        self.lconvlayer4 = nn.Conv3d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = nn.BatchNorm3d(128)

        self.lconvlayer5 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 4,10

        self.lconvlayer6 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = nn.BatchNorm3d(128)

        self.lconvlayer7 = nn.Conv3d(
            128, 256, kernel_size=4, stride=2, padding=1
        )  # 2, 5

        self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer7_bn = nn.BatchNorm3d(128)

        self.rconvTlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)

        self.rconvTlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5_bn = nn.BatchNorm3d(128)

        self.rconvTlayer4 = nn.Conv3d(128, 64, kernel_size=5, padding=2)
        self.rconvlayer4 = nn.Conv3d(64, 64, kernel_size=5, padding=2)

        self.rconvTlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer3_bn = nn.BatchNorm3d(64)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.rconvlayer2 = nn.Conv3d(32, 32, kernel_size=5, padding=2)

        self.rconvTlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        #        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvlayer1_bn = nn.BatchNorm3d(32)

        self.out_layer = nn.Conv3d(32, 1, kernel_size=1, stride=1)

    def forward(self, x):
        #        xlact = self.act1(x)
        #        xl0 = self.relu(self.in_layer_bn(self.in_layer(xlact)))
        xl0 = self.relu(self.in_layer_bn(self.in_layer(x)))
        xl1 = self.relu(self.lconvlayer1(xl0))
        xl2 = self.relu(self.lconvlayer2_bn(self.lconvlayer2(xl1)))
        xl3 = self.relu(self.lconvlayer3(xl2))
        xl4 = self.relu(self.lconvlayer4_bn(self.lconvlayer4(xl3)))
        xl5 = self.relu(self.lconvlayer5(xl4))
        xl6 = self.relu(self.lconvlayer6_bn(self.lconvlayer6(xl5)))

        xr6 = xl6
        xr61 = self.relu(
            self.rconvTlayer6(
                self.upsample(self.relu(self.rconvlayer7_bn(self.rconvlayer7(xr6))))
            )
        )

        xr51 = torch.add(xr61, xl5)
        xr5 = self.relu(
            self.rconvTlayer5(self.upsample(self.relu(self.rconvlayer6(xr51))))
        )
        xr4 = torch.add(xr5, xl4)
        xr4 = self.relu(
            self.rconvTlayer4(
                self.upsample(self.relu(self.rconvlayer5_bn(self.rconvlayer5(xr4))))
            )
        )

        xr3 = torch.add(xr4, xl3)
        xr3 = self.relu(
            self.rconvTlayer3(self.upsample(self.relu(self.rconvlayer4(xr3))))
        )

        xr2 = torch.add(xr3, xl2)
        xr2 = self.relu(
            self.rconvTlayer2(
                self.upsamplez(self.relu(self.rconvlayer3_bn(self.rconvlayer3(xr2))))
            )
        )

        xr1 = torch.add(xr2, xl1)
        xr1 = self.relu(
            self.rconvTlayer1(self.upsamplez(self.relu(self.rconvlayer2(xr1))))
        )

        xr0 = torch.add(xr1, xl0)

        xr0 = self.relu(self.rconvlayer1_bn(self.rconvlayer1(xr0)))
        out_layer = self.out_layer(xr0)

        return out_layer


class SKUNET100_DICE_BN_NEW_MOD4_EX3(nn.Module):
    def __init__(self):
        super(SKUNET100_DICE_BN_NEW_MOD4_EX3, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        #        self.act1 = my_act()

        self.in_layer = nn.Conv3d(1, 32, kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = InPlaceABN(32)

        self.lconvlayer1 = nn.Conv3d(
            32, 32, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,160
        self.lconvlayer1_bn = InPlaceABN(32)

        self.lconvlayer2 = nn.Conv3d(
            32, 64, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,80
        self.lconvlayer2_bn = InPlaceABN(64)

        self.lconvlayer3 = nn.Conv3d(
            64, 64, kernel_size=4, stride=2, padding=1
        )  # 16,40
        self.lconvlayer3_bn = InPlaceABN(64)

        self.lconvlayer4 = nn.Conv3d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = InPlaceABN(128)

        self.lconvlayer5 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 4,10
        self.lconvlayer5_bn = InPlaceABN(128)

        self.lconvlayer6 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = InPlaceABN(128)

        # self.lconvlayer7 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7_bn = InPlaceABN(256)
        #
        # self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer7_bn = InPlaceABN(128)

        self.rconvTlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvTlayer6_bn = InPlaceABN(128)
        self.rconvlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer6_bn = InPlaceABN(128)

        self.rconvTlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvTlayer5_bn = InPlaceABN(128)
        self.rconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5_bn = InPlaceABN(128)

        self.rconvTlayer4 = nn.Conv3d(128, 64, kernel_size=5, padding=2)
        self.rconvTlayer4_bn = InPlaceABN(64)
        self.rconvlayer4 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer4_bn = InPlaceABN(64)

        self.rconvTlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvTlayer3_bn = InPlaceABN(64)
        self.rconvlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer3_bn = InPlaceABN(64)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.rconvTlayer2_bn = InPlaceABN(32)
        self.rconvlayer2 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvlayer2_bn = InPlaceABN(32)

        self.rconvTlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvTlayer1_bn = InPlaceABN(32)
        #        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvlayer1_bn = InPlaceABN(32)

        self.out_layer = nn.Conv3d(32, 1, kernel_size=1, stride=1)

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

        return out_layer


class SKUNET100_DICE_BN_NEW_MOD4_EX4(nn.Module):
    def __init__(self):
        super(SKUNET100_DICE_BN_NEW_MOD4_EX4, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        #        self.act1 = my_act()

        self.in_layer = nn.Conv3d(1, 32, kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = InPlaceABNSync(32)

        self.lconvlayer1 = nn.Conv3d(
            32, 32, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,160
        self.lconvlayer1_bn = InPlaceABNSync(32)

        self.lconvlayer2 = nn.Conv3d(
            32, 64, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,80
        self.lconvlayer2_bn = InPlaceABNSync(64)

        self.lconvlayer3 = nn.Conv3d(
            64, 64, kernel_size=4, stride=2, padding=1
        )  # 16,40
        self.lconvlayer3_bn = InPlaceABNSync(64)

        self.lconvlayer4 = nn.Conv3d(
            64, 128, kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = InPlaceABNSync(128)

        self.lconvlayer5 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 4,10
        self.lconvlayer5_bn = InPlaceABNSync(128)

        self.lconvlayer6 = nn.Conv3d(
            128, 128, kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = InPlaceABNSync(128)

        # self.lconvlayer7 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7_bn = InPlaceABN(256)
        #
        # self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer7_bn = InPlaceABNSync(128)

        self.rconvTlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvTlayer6_bn = InPlaceABNSync(128)
        self.rconvlayer6 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer6_bn = InPlaceABNSync(128)

        self.rconvTlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvTlayer5_bn = InPlaceABNSync(128)
        self.rconvlayer5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.rconvlayer5_bn = InPlaceABNSync(128)

        self.rconvTlayer4 = nn.Conv3d(128, 64, kernel_size=5, padding=2)
        self.rconvTlayer4_bn = InPlaceABNSync(64)
        self.rconvlayer4 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer4_bn = InPlaceABNSync(64)

        self.rconvTlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvTlayer3_bn = InPlaceABNSync(64)
        self.rconvlayer3 = nn.Conv3d(64, 64, kernel_size=5, padding=2)
        self.rconvlayer3_bn = InPlaceABNSync(64)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.rconvTlayer2_bn = InPlaceABNSync(32)
        self.rconvlayer2 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvlayer2_bn = InPlaceABNSync(32)

        self.rconvTlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvTlayer1_bn = InPlaceABNSync(32)
        #        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvlayer1 = nn.Conv3d(32, 32, kernel_size=5, padding=2)
        self.rconvlayer1_bn = InPlaceABNSync(32)

        self.out_layer = nn.Conv3d(32, 1, kernel_size=1, stride=1)

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

        return out_layer


class SKUNET100_DICE_BN_NEW_MOD4_EX5(nn.Module):
    def __init__(self):
        super(SKUNET100_DICE_BN_NEW_MOD4_EX5, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear")
        self.upsamplez = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear")
        #        self.act1 = my_act()

        self.in_layer = nn.Conv3d(1, 48, kernel_size=5, padding=2)  # 32,320
        self.in_layer_bn = InPlaceABNSync(48)

        self.lconvlayer1 = nn.Conv3d(
            48, 48, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,160
        self.lconvlayer1_bn = InPlaceABNSync(48)

        self.lconvlayer2 = nn.Conv3d(
            48, 96, kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2)
        )  # 32,80
        self.lconvlayer2_bn = InPlaceABNSync(96)

        self.lconvlayer3 = nn.Conv3d(
            96, 96, kernel_size=4, stride=2, padding=1
        )  # 16,40
        self.lconvlayer3_bn = InPlaceABNSync(96)

        self.lconvlayer4 = nn.Conv3d(
            96, 192, kernel_size=4, stride=2, padding=1
        )  # 8,20
        self.lconvlayer4_bn = InPlaceABNSync(192)

        self.lconvlayer5 = nn.Conv3d(
            192, 192, kernel_size=4, stride=2, padding=1
        )  # 4,10
        self.lconvlayer5_bn = InPlaceABNSync(192)

        self.lconvlayer6 = nn.Conv3d(
            192, 192, kernel_size=4, stride=2, padding=1
        )  # 2, 5
        self.lconvlayer6_bn = InPlaceABNSync(192)

        # self.lconvlayer7 = nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1)  # 2, 5
        # self.lconvlayer7_bn = InPlaceABN(256)
        #
        # self.rconvTlayer7 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.rconvlayer7 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.rconvlayer7_bn = InPlaceABNSync(192)

        self.rconvTlayer6 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.rconvTlayer6_bn = InPlaceABNSync(192)
        self.rconvlayer6 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.rconvlayer6_bn = InPlaceABNSync(192)

        self.rconvTlayer5 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.rconvTlayer5_bn = InPlaceABNSync(192)
        self.rconvlayer5 = nn.Conv3d(192, 192, kernel_size=3, padding=1)
        self.rconvlayer5_bn = InPlaceABNSync(192)

        self.rconvTlayer4 = nn.Conv3d(192, 96, kernel_size=5, padding=2)
        self.rconvTlayer4_bn = InPlaceABNSync(96)
        self.rconvlayer4 = nn.Conv3d(96, 96, kernel_size=5, padding=2)
        self.rconvlayer4_bn = InPlaceABNSync(96)

        self.rconvTlayer3 = nn.Conv3d(96, 96, kernel_size=5, padding=2)
        self.rconvTlayer3_bn = InPlaceABNSync(96)
        self.rconvlayer3 = nn.Conv3d(96, 96, kernel_size=5, padding=2)
        self.rconvlayer3_bn = InPlaceABNSync(96)

        #        self.rconvTlayer2 = nn.ConvTranspose3d(64, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvTlayer2 = nn.Conv3d(96, 48, kernel_size=5, padding=2)
        self.rconvTlayer2_bn = InPlaceABNSync(48)
        self.rconvlayer2 = nn.Conv3d(48, 48, kernel_size=5, padding=2)
        self.rconvlayer2_bn = InPlaceABNSync(48)

        self.rconvTlayer1 = nn.Conv3d(48, 48, kernel_size=5, padding=2)
        self.rconvTlayer1_bn = InPlaceABNSync(48)
        #        self.rconvTlayer1 = nn.ConvTranspose3d(32, 32, kernel_size = (1,2,2), stride = (1,2,2))
        self.rconvlayer1 = nn.Conv3d(48, 48, kernel_size=5, padding=2)
        self.rconvlayer1_bn = InPlaceABNSync(48)

        self.out_layer = nn.Conv3d(48, 1, kernel_size=1, stride=1)

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

        return out_layer
        return out_layer
