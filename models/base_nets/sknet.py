from base_nets.layer import *


def make_layer(block, inplanes, planes, blocks):
    layers = []
    stride = 1
    downsample = None
    if planes != inplanes:
        downsample = nn.Conv3d(inplanes, planes, kernel_size=1)
    layers.append(block(inplanes, planes, stride, downsample=downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))
    return nn.Sequential(*layers)


class Unet1(nn.Module):
    def __init__(self, feats, block, n_blocks):
        super(Unet1, self).__init__()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")

        self.stem = BasicConvBN3d(1, feats[0], (3, 5, 5), padding=(1, 2, 2))

        self.layer1 = make_layer(block, feats[0], feats[1], n_blocks)
        self.layer2 = make_layer(block, feats[1], feats[2], n_blocks)
        self.layer3 = make_layer(block, feats[2], feats[3], n_blocks)
        self.layer4 = make_layer(block, feats[3], feats[4], n_blocks)

        self.layer1b = make_layer(block, feats[1], feats[0], 1)
        self.layer2b = make_layer(block, feats[2], feats[1], 1)
        self.layer3b = make_layer(block, feats[3], feats[2], 1)
        self.layer4b = make_layer(block, feats[4], feats[3], 1)

        self.final = nn.Conv3d(feats[0], 2, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x = self.pool1(x0)
        x1 = self.layer1(x)
        x = self.pool(x1)
        x2 = self.layer2(x)
        x = self.pool(x2)
        x3 = self.layer3(x)
        x = self.pool(x3)
        x4 = self.layer4(x)

        x = self.layer4b(x4)
        x = F.interpolate(x, scale_factor=2) + x3
        x = self.layer3b(x)
        x = F.interpolate(x, scale_factor=2) + x2
        x = self.layer2b(x)
        x = F.interpolate(x, scale_factor=2) + x1
        x = self.layer1b(x)
        x = F.interpolate(x, scale_factor=(1, 2, 2)) + x0
        x = self.final(x)
        return x


class toynet(Unet1):
    def __init__(self):
        super(toynet, self).__init__(
            feats=[24, 48, 48, 64, 64], block=ResBlock, n_blocks=2
        )


class toynet2(Unet1):
    def __init__(self):
        super(toynet2, self).__init__(
            feats=[24, 48, 48, 64, 64], block=Basic_block_inplace, n_blocks=2
        )


class toynet3(Unet1):
    def __init__(self):
        super(toynet3, self).__init__(
            feats=[24, 48, 48, 64, 64], block=ResBlockGN, n_blocks=2
        )
