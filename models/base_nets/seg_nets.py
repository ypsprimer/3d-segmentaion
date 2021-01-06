from base_nets.layer import *


###############################################################################################
########################################## Unet模型 ###########################################
class UNET(nn.Module):
    def __init__(self):
        super(UNET, self).__init__()
        self.preBlock1 = nn.Sequential(
            BasicConvBNS3d(1, 32, kernel_size=3, padding=1),
            BasicConvBNS3d(32, 64, kernel_size=3, padding=1),
        )
        self.DownBlock1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock2 = nn.Sequential(
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 128, kernel_size=3, padding=1),
        )
        self.DownBlock2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock3 = nn.Sequential(
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 256, kernel_size=3, padding=1),
        )
        self.DownBlock3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock4 = nn.Sequential(
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 512, kernel_size=3, padding=1),
        )
        self.preConvTranspose1 = BasicConvTransposeBNS3d(
            512, 512, kernel_size=2, stride=2
        )
        self.preBlock5 = nn.Sequential(
            BasicConvBNS3d(768, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
        )
        self.preConvTranspose2 = BasicConvTransposeBNS3d(
            256, 256, kernel_size=2, stride=2
        )
        self.preBlock6 = nn.Sequential(
            BasicConvBNS3d(384, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
        )
        self.preConvTranspose3 = BasicConvTransposeBNS3d(
            128, 128, kernel_size=2, stride=2
        )
        self.preBlock7 = nn.Sequential(
            BasicConvBNS3d(192, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
        )
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(64, 3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.preBlock1(x)
        x0 = self.DownBlock1(x)
        x0 = self.preBlock2(x0)
        x1 = self.DownBlock2(x0)
        x1 = self.preBlock3(x1)
        x2 = self.DownBlock3(x1)
        x2 = self.preBlock4(x2)
        x3 = self.preConvTranspose1(x2)
        x4 = torch.cat((x3, x1), 1)
        x4 = self.preBlock5(x4)
        x5 = self.preConvTranspose2(x4)
        x6 = torch.cat((x5, x0), 1)
        x6 = self.preBlock6(x6)
        x7 = self.preConvTranspose3(x6)
        x8 = torch.cat((x7, x), 1)
        x8 = self.preBlock7(x8)
        x8 = self.drop(x8)
        out = self.FCN_out(x8)
        return self.softmax(out)


class UNET_HALF(nn.Module):
    def __init__(self):
        super(UNET_HALF, self).__init__()
        self.preBlock1 = nn.Sequential(
            BasicConvBNS3d(1, 16, kernel_size=3, padding=1),
            BasicConvBNS3d(16, 32, kernel_size=3, padding=1),
        )
        self.DownBlock1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock2 = nn.Sequential(
            BasicConvBNS3d(32, 32, kernel_size=3, padding=1),
            BasicConvBNS3d(32, 64, kernel_size=3, padding=1),
        )
        self.DownBlock2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock3 = nn.Sequential(
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 128, kernel_size=3, padding=1),
        )
        self.DownBlock3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock4 = nn.Sequential(
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 256, kernel_size=3, padding=1),
        )
        self.preConvTranspose1 = BasicConvTransposeBNS3d(
            256, 256, kernel_size=2, stride=2
        )
        self.preBlock5 = nn.Sequential(
            BasicConvBNS3d(384, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
        )
        self.preConvTranspose2 = BasicConvTransposeBNS3d(
            128, 128, kernel_size=2, stride=2
        )
        self.preBlock6 = nn.Sequential(
            BasicConvBNS3d(192, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
        )
        self.preConvTranspose3 = BasicConvTransposeBNS3d(
            64, 64, kernel_size=2, stride=2
        )
        self.preBlock7 = nn.Sequential(
            BasicConvBNS3d(96, 32, kernel_size=3, padding=1),
            BasicConvBNS3d(32, 16, kernel_size=3, padding=1),
        )
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(16, 1, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.preBlock1(x)
        x0 = self.DownBlock1(x)
        x0 = self.preBlock2(x0)
        x1 = self.DownBlock2(x0)
        x1 = self.preBlock3(x1)
        x2 = self.DownBlock3(x1)
        x2 = self.preBlock4(x2)
        x3 = self.preConvTranspose1(x2)
        x4 = torch.cat((x3, x1), 1)
        x4 = self.preBlock5(x4)
        x5 = self.preConvTranspose2(x4)
        x6 = torch.cat((x5, x0), 1)
        x6 = self.preBlock6(x6)
        x7 = self.preConvTranspose3(x6)
        x8 = torch.cat((x7, x), 1)
        x8 = self.preBlock7(x8)
        x8 = self.drop(x8)
        out = self.FCN_out(x8)
        return out


class UNET_4class(nn.Module):
    def __init__(self):
        super(UNET_4class, self).__init__()
        self.preBlock1 = nn.Sequential(
            BasicConvBNS3d(1, 32, kernel_size=3, padding=1),
            BasicConvBNS3d(32, 64, kernel_size=3, padding=1),
        )
        self.DownBlock1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock2 = nn.Sequential(
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 128, kernel_size=3, padding=1),
        )
        self.DownBlock2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock3 = nn.Sequential(
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 256, kernel_size=3, padding=1),
        )
        self.DownBlock3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.preBlock4 = nn.Sequential(
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 512, kernel_size=3, padding=1),
        )
        self.preConvTranspose1 = BasicConvTransposeBNS3d(
            512, 512, kernel_size=2, stride=2
        )
        self.preBlock5 = nn.Sequential(
            BasicConvBNS3d(768, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
        )
        self.preConvTranspose2 = BasicConvTransposeBNS3d(
            256, 256, kernel_size=2, stride=2
        )
        self.preBlock6 = nn.Sequential(
            BasicConvBNS3d(384, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
        )
        self.preConvTranspose3 = BasicConvTransposeBNS3d(
            128, 128, kernel_size=2, stride=2
        )
        self.preBlock7 = nn.Sequential(
            BasicConvBNS3d(192, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
        )
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(64, 4, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x = self.preBlock1(x)
        x0 = self.DownBlock1(x)
        x0 = self.preBlock2(x0)
        x1 = self.DownBlock2(x0)
        x1 = self.preBlock3(x1)
        x2 = self.DownBlock3(x1)
        x2 = self.preBlock4(x2)
        x3 = self.preConvTranspose1(x2)
        x4 = torch.cat((x3, x1), 1)
        x4 = self.preBlock5(x4)
        x5 = self.preConvTranspose2(x4)
        x6 = torch.cat((x5, x0), 1)
        x6 = self.preBlock6(x6)
        x7 = self.preConvTranspose3(x6)
        x8 = torch.cat((x7, x), 1)
        x8 = self.preBlock7(x8)
        x8 = self.drop(x8)
        out = self.FCN_out(x8)
        return self.softmax(out)


###############################################################################################################
####################################### 原始Vnet+inplaceabn+Sync模型：5*5 #####################################
def passthrough(x, **kwargs):
    return x


class LUConvABNS(nn.Module):  # 一个conv -> bn -> relu 的标准组合，信道数保持不变
    def __init__(self, nchan):
        super(LUConvABNS, self).__init__()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn = InPlaceABNSync(nchan)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LUConvABN(nn.Module):  # 一个conv -> bn -> relu 的标准组合，信道数保持不变
    def __init__(self, nchan):
        super(LUConvABN, self).__init__()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn = InPlaceABN(nchan)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class LUConv(nn.Module):  # 一个conv -> bn -> relu 的标准组合，信道数保持不变
    def __init__(self, nchan):
        super(LUConv, self).__init__()
        self.conv = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(nchan)
        self.relu = nn.PReLU(nchan)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.bn(x))
        return x


def _make_nConvABNS(nchan, depth):  # 多个conv -> bn -> relu 的标准组合，信道数保持不变
    layers = []
    for _ in range(depth - 1):
        layers.append(LUConvABNS(nchan))
    return nn.Sequential(*layers)


def _make_nConvABN(nchan, depth):  # 多个conv -> bn -> relu 的标准组合，信道数保持不变
    layers = []
    for _ in range(depth - 1):
        layers.append(LUConvABN(nchan))
    return nn.Sequential(*layers)


def _make_nConv(nchan, depth):  # 多个conv -> bn -> relu 的标准组合，信道数保持不变
    layers = []
    for _ in range(depth - 1):
        layers.append(LUConv(nchan))
    return nn.Sequential(*layers)


class InputTransitionABNS(nn.Module):  # 设置输入/输出信道数，
    def __init__(self, inChans, outChans):
        super(InputTransitionABNS, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn = InPlaceABNSync(outChans)
        self.relu = nn.PReLU(outChans)
        self.out = outChans

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.bn(self.conv(x))
        # split input in to 16 channels
        x = torch.unsqueeze(torch.sum(x, 1), dim=1)
        if self.out == 16:
            x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        else:
            x16 = torch.cat(
                (
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                ),
                1,
            )
        out = self.relu(torch.add(out, x16))
        return out


class InputTransitionABN(nn.Module):  # 设置输入/输出信道数，
    def __init__(self, inChans, outChans):
        super(InputTransitionABN, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn = InPlaceABN(outChans)
        self.relu = nn.PReLU(outChans)
        self.out = outChans

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu(self.bn(self.conv(x)))
        # split input in to 16 channels
        x = torch.unsqueeze(torch.sum(x, 1), dim=1)
        if self.out == 16:
            x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        else:
            x16 = torch.cat(
                (
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                ),
                1,
            )
        out = self.relu(torch.add(out, x16))
        return out


class InputTransition(nn.Module):  # 设置输入/输出信道数，
    def __init__(self, inChans, outChans):
        super(InputTransition, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(outChans)
        self.relu = nn.PReLU(outChans)
        self.out = outChans

    def forward(self, x):
        # do we want a PRELU here as well?
        out = self.relu(self.bn(self.conv(x)))
        # split input in to 16 channels
        x = torch.unsqueeze(torch.sum(x, 1), dim=1)
        if self.out == 16:
            x16 = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), 1)
        else:
            x16 = torch.cat(
                (
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                    x,
                ),
                1,
            )
        out = self.relu(torch.add(out, x16))
        return out


class DownTransitionABNS(nn.Module):  # 一般情况：输入信道->2*输入信道， 可以设置层数
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransitionABNS, self).__init__()
        self.down_conv = nn.Conv3d(
            inChans, outChans, kernel_size=2, stride=2
        )  # 卷积降采样改变信道数，后面跟bn和relu
        self.bn = InPlaceABNSync(outChans)
        self.do = passthrough  # 降采样的输出后是否要dropout
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do = nn.Dropout3d()
        self.ops = _make_nConvABNS(outChans, nConvs)

    def forward(self, x):
        down = self.bn(self.down_conv(x))
        out = self.do(down)
        out = self.ops(out)
        out = self.relu(torch.add(out, down))
        return out


class DownTransitionABN(nn.Module):  # 一般情况：输入信道->2*输入信道， 可以设置层数
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransitionABN, self).__init__()
        self.down_conv = nn.Conv3d(
            inChans, outChans, kernel_size=2, stride=2
        )  # 卷积降采样改变信道数，后面跟bn和relu
        self.bn = InPlaceABN(outChans)
        self.do = passthrough  # 降采样的输出后是否要dropout
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do = nn.Dropout3d()
        self.ops = _make_nConvABN(outChans, nConvs)

    def forward(self, x):
        down = self.bn(self.down_conv(x))
        out = self.do(down)
        out = self.ops(out)
        out = self.relu(torch.add(out, down))
        return out


class DownTransition(nn.Module):  # 一般情况：输入信道->2*输入信道， 可以设置层数
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(DownTransition, self).__init__()
        self.down_conv = nn.Conv3d(
            inChans, outChans, kernel_size=2, stride=2
        )  # 卷积降采样改变信道数，后面跟bn和relu
        self.bn = nn.BatchNorm3d(outChans)
        self.do = passthrough  # 降采样的输出后是否要dropout
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x):
        down = self.relu(self.bn(self.down_conv(x)))
        out = self.do(down)
        out = self.ops(out)
        out = self.relu(torch.add(out, down))
        return out


class UpTransitionABNS(nn.Module):  # 设置输入输出信道数，可设置层数，同时具有组合信道的功能
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransitionABNS, self).__init__()
        self.up_conv = nn.ConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2
        )  # 256-128
        self.bn1 = InPlaceABNSync(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConvABNS(outChans, nConvs)

    def forward(self, x, skipx):
        x = self.do1(x)  # 256
        skipxdo = self.do2(skipx)  # 128
        x = self.bn1(self.up_conv(x))  # 128
        xcat = torch.cat((x, skipxdo), 1)  # 256
        out = self.ops(xcat)  # 256
        out = self.relu(torch.add(out, xcat))
        return out


class UpTransitionABN(nn.Module):  # 设置输入输出信道数，可设置层数，同时具有组合信道的功能
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransitionABN, self).__init__()
        self.up_conv = nn.ConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2
        )  # 256-128
        self.bn1 = InPlaceABN(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConvABN(outChans, nConvs)

    def forward(self, x, skipx):
        x = self.do1(x)  # 256
        skipxdo = self.do2(skipx)  # 128
        x = self.bn1(self.up_conv(x))  # 128
        xcat = torch.cat((x, skipxdo), 1)  # 256
        out = self.ops(xcat)  # 256
        out = self.relu(torch.add(out, xcat))
        return out


class UpTransition(nn.Module):  # 设置输入输出信道数，可设置层数，同时具有组合信道的功能
    def __init__(self, inChans, outChans, nConvs, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv1 = nn.ConvTranspose3d(
            inChans, outChans // 2, kernel_size=2, stride=2
        )  # 256-128
        self.bn1 = nn.BatchNorm3d(outChans // 2)
        self.relu1 = nn.PReLU(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu = nn.PReLU(outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs)

    def forward(self, x, skipx):
        x = self.do1(x)  # 256
        skipxdo = self.do2(skipx)  # 128
        x = self.relu1(self.bn1(self.up_conv1(x)))  # 128
        xcat = torch.cat((x, skipxdo), 1)  # 256
        out = self.ops(xcat)  # 256
        out = self.relu(torch.add(out, xcat))
        return out


class OutputTransitionABNS(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransitionABNS, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn = InPlaceABNSync(outChans)
        self.softmax = nn.Softmax(1)
        self.out = outChans

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.out != 1:
            out = self.softmax(out)
        return out


class OutputTransitionABN(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransitionABN, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn = InPlaceABN(outChans)
        self.softmax = nn.Softmax(1)
        self.out = outChans

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.out != 1:
            out = self.softmax(out)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, outChans):
        super(OutputTransition, self).__init__()
        self.conv = nn.Conv3d(inChans, outChans, kernel_size=1)
        self.bn = nn.BatchNorm3d(outChans)
        self.relu = nn.PReLU(outChans)
        self.softmax = nn.Softmax(1)
        self.out = outChans

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        if self.out != 1:
            out = self.softmax(out)
        return out


class VNET_BNS(nn.Module):
    def __init__(self, elu=True):
        super(VNET_BNS, self).__init__()
        self.in_tr = InputTransitionABNS(1, 16)
        self.down_tr32 = DownTransitionABNS(16, 32, 2)
        self.down_tr64 = DownTransitionABNS(32, 64, 3)
        self.down_tr128 = DownTransitionABNS(64, 128, 3)
        self.down_tr256 = DownTransitionABNS(128, 256, 3)
        self.up_tr256 = UpTransitionABNS(256, 256, 3)
        self.up_tr128 = UpTransitionABNS(256, 128, 3)
        self.up_tr64 = UpTransitionABNS(128, 64, 2)
        self.up_tr32 = UpTransitionABNS(64, 32, 1)
        self.out_tr = OutputTransitionABNS(32, 3)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class VNET_BN(nn.Module):
    def __init__(self, elu=True):
        super(VNET_BN, self).__init__()
        self.in_tr = InputTransitionABN(1, 16)
        self.down_tr32 = DownTransitionABN(16, 32, 2)
        self.down_tr64 = DownTransitionABN(32, 64, 3)
        self.down_tr128 = DownTransitionABN(64, 128, 3)
        self.down_tr256 = DownTransitionABN(128, 256, 3)
        self.up_tr256 = UpTransitionABN(256, 256, 3)
        self.up_tr128 = UpTransitionABN(256, 128, 3)
        self.up_tr64 = UpTransitionABN(128, 64, 2)
        self.up_tr32 = UpTransitionABN(64, 32, 1)
        self.out_tr = OutputTransitionABN(32, 3)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


class VNET_Normal(nn.Module):
    def __init__(self, elu=True):
        super(VNET_Normal, self).__init__()
        self.in_tr = InputTransition(1, 16)
        self.down_tr32 = DownTransition(16, 32, 2)
        self.down_tr64 = DownTransition(32, 64, 3)
        self.down_tr128 = DownTransition(64, 128, 3)
        self.down_tr256 = DownTransition(128, 256, 3)
        self.up_tr256 = UpTransition(256, 256, 3)
        self.up_tr128 = UpTransition(256, 128, 3)
        self.up_tr64 = UpTransition(128, 64, 2)
        self.up_tr32 = UpTransition(64, 32, 1)
        self.out_tr = OutputTransition(32, 3)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


######################################################################
######################################################################


class UNET_DownConv(nn.Module):
    def __init__(self):
        super(UNET_DownConv, self).__init__()
        self.preBlock1 = nn.Sequential(
            BasicConvBNS3d(1, 32, kernel_size=3, padding=1),
            BasicConvBNS3d(32, 64, kernel_size=3, padding=1),
        )
        self.DownBlock1 = DownBlockBNS(64, 64, kernel_size=2, stride=2)
        self.preBlock2 = nn.Sequential(
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 128, kernel_size=3, padding=1),
        )
        self.DownBlock2 = DownBlockBNS(128, 128, kernel_size=2, stride=2)
        self.preBlock3 = nn.Sequential(
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 256, kernel_size=3, padding=1),
        )
        self.DownBlock3 = DownBlockBNS(256, 256, kernel_size=2, stride=2)
        self.preBlock4 = nn.Sequential(
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 512, kernel_size=3, padding=1),
        )
        self.preConvTranspose1 = BasicConvTransposeBNS3d(
            512, 512, kernel_size=2, stride=2
        )
        self.preBlock5 = nn.Sequential(
            BasicConvBNS3d(768, 256, kernel_size=3, padding=1),
            BasicConvBNS3d(256, 256, kernel_size=3, padding=1),
        )
        self.preConvTranspose2 = BasicConvTransposeBNS3d(
            256, 256, kernel_size=2, stride=2
        )
        self.preBlock6 = nn.Sequential(
            BasicConvBNS3d(384, 128, kernel_size=3, padding=1),
            BasicConvBNS3d(128, 128, kernel_size=3, padding=1),
        )
        self.preConvTranspose3 = BasicConvTransposeBNS3d(
            128, 128, kernel_size=2, stride=2
        )
        self.preBlock7 = nn.Sequential(
            BasicConvBNS3d(192, 64, kernel_size=3, padding=1),
            BasicConvBNS3d(64, 64, kernel_size=3, padding=1),
        )
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(64, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.preBlock1(x)
        x0 = self.DownBlock1(x)
        x0 = self.preBlock2(x0)
        x1 = self.DownBlock2(x0)
        x1 = self.preBlock3(x1)
        x2 = self.DownBlock3(x1)
        x2 = self.preBlock4(x2)
        x3 = self.preConvTranspose1(x2)
        x4 = torch.cat((x3, x1), 1)
        x4 = self.preBlock5(x4)
        x5 = self.preConvTranspose2(x4)
        x6 = torch.cat((x5, x0), 1)
        x6 = self.preBlock6(x6)
        x7 = self.preConvTranspose3(x6)
        x8 = torch.cat((x7, x), 1)
        x8 = self.preBlock7(x8)
        x8 = self.drop(x8)
        out = self.FCN_out(x8)
        return out


class SKRUNET(nn.Module):
    def __init__(self, in_planes):
        super(SKRUNET, self).__init__()
        self.preBlock1 = nn.Sequential(
            BasicConvBNS3d(in_planes, 24, kernel_size=3, padding=1)
        )
        self.DownBlock1 = DownBlockBNS(24, 48, kernel_size=2, stride=2)
        self.preBlock2 = nn.Sequential(
            ResBlockBN(48, 48), ResBlockBN(48, 48), ResBlockBN(48, 48)
        )
        self.DownBlock2 = DownBlockBNS(48, 48, kernel_size=2, stride=2)
        self.preBlock3 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.DownBlock3 = DownBlockBNS(48, 48, kernel_size=2, stride=2)
        self.preBlock4 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.DownBlock4 = DownBlockBNS(48, 48, kernel_size=2, stride=2)
        self.preBlock5 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.DownBlock5 = DownBlockBNS(48, 48, kernel_size=2, stride=2)
        self.preBlock6 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.DownBlock6 = DownBlockBNS(48, 48, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.preBlock7 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.preConvTranspose1 = BasicConvTransposeBNS3d(
            48, 48, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.preBlock8 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.preConvTranspose2 = BasicConvTransposeBNS3d(
            48, 48, kernel_size=2, stride=2
        )
        self.preBlock9 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.preConvTranspose3 = BasicConvTransposeBNS3d(
            48, 48, kernel_size=2, stride=2
        )
        self.preBlock10 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.preConvTranspose4 = BasicConvTransposeBNS3d(
            48, 48, kernel_size=2, stride=2
        )
        self.preBlock11 = nn.Sequential(
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
            ResBlockBN(48, 48),
        )
        self.preConvTranspose5 = BasicConvTransposeBNS3d(
            48, 48, kernel_size=2, stride=2
        )
        self.preBlock12 = nn.Sequential(
            ResBlockBN(48, 48), ResBlockBN(48, 48), ResBlockBN(48, 48)
        )
        self.preConvTranspose6 = BasicConvTransposeBNS3d(
            48, 24, kernel_size=2, stride=2
        )
        self.preBlock13 = nn.Sequential(ResBlockBN(24, 24))
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(24, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.preBlock1(x)
        x0 = self.DownBlock1(x)
        x0 = self.preBlock2(x0)
        x1 = self.DownBlock2(x0)
        x1 = self.preBlock3(x1)
        x2 = self.DownBlock3(x1)
        x2 = self.preBlock4(x2)
        x3 = self.DownBlock4(x2)
        x3 = self.preBlock5(x3)
        x4 = self.DownBlock5(x3)
        x4 = self.preBlock6(x4)
        x5 = self.DownBlock6(x4)
        x5 = self.preBlock7(x5)
        x6 = self.preConvTranspose1(x5)
        x7 = x6 + x4
        x7 = self.preBlock8(x7)
        x8 = self.preConvTranspose2(x7)
        x9 = x8 + x3
        x9 = self.preBlock9(x9)
        x10 = self.preConvTranspose3(x9)
        x11 = x10 + x2
        x11 = self.preBlock10(x11)
        x12 = self.preConvTranspose4(x11)
        x13 = x12 + x1
        x13 = self.preBlock11(x13)
        x14 = self.preConvTranspose5(x13)
        x15 = x14 + x0
        x15 = self.preBlock12(x15)
        x16 = self.preConvTranspose6(x15)
        x17 = x16 + x
        x17 = self.preBlock13(x17)
        x17 = self.drop(x17)
        out = self.FCN_out(x17)
        return out


#################################################################################################
##################################  FC-RESNET模型 ###############################################
class SimpleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout=False):
        super(SimpleBlock, self).__init__()
        self.bn1 = InPlaceABNSync(in_planes)
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=3, padding=1)
        self.bn2 = InPlaceABNSync(in_planes)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout3d(p=0.5)
        if in_planes != out_planes:
            self.respath = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1)
        self.drop = dropout
        self.in_planes = in_planes
        self.out_planes = out_planes

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        if self.drop == True:
            out = self.dropout(out)
        if self.in_planes != self.out_planes:
            residual = self.respath(residual)
        out += residual
        return out


class Bottleneckblock(nn.Module):
    def __init__(self, in_planes, out_planes, dropout=False):
        super(Bottleneckblock, self).__init__()
        self.bn1 = InPlaceABNSync(in_planes)
        self.conv1 = nn.Conv3d(in_planes, in_planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = InPlaceABNSync(in_planes)
        self.conv2 = nn.Conv3d(in_planes, in_planes, kernel_size=3, stride=1, padding=1)
        self.bn3 = InPlaceABNSync(in_planes)
        self.conv3 = nn.Conv3d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0
        )
        self.dropout = nn.Dropout3d(p=0.5)
        if in_planes != out_planes:
            self.respath = nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1)
        self.drop = dropout
        self.in_planes = in_planes
        self.out_planes = out_planes

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.conv3(out)
        if self.drop == True:
            out = self.dropout(out)
        if self.in_planes != self.out_planes:
            residual = self.respath(residual)
        out += residual
        return out


class FCRESNET_1t3(nn.Module):
    def __init__(self):
        super(FCRESNET_1t3, self).__init__()
        self.downBlock1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.downBlock2 = nn.Sequential(
            SimpleBlock(32, 32), SimpleBlock(32, 32), SimpleBlock(32, 32)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.downBlock3 = nn.Sequential(
            Bottleneckblock(32, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.downBlock4 = nn.Sequential(
            Bottleneckblock(64, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.across = nn.Sequential(
            Bottleneckblock(128, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 256),
            Bottleneckblock(256, 128),
        )
        self.upsample1 = BasicConvTransposeBNS3d(128, 128, kernel_size=2, stride=2)
        self.upBlock1 = nn.Sequential(
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 128),
            Bottleneckblock(128, 64),
        )
        self.upsample2 = BasicConvTransposeBNS3d(64, 64, kernel_size=2, stride=2)
        self.upBlock2 = nn.Sequential(
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 64),
            Bottleneckblock(64, 32),
        )
        self.upsample3 = BasicConvTransposeBNS3d(32, 32, kernel_size=2, stride=2)
        self.upBlock3 = nn.Sequential(
            SimpleBlock(32, 32), SimpleBlock(32, 32), SimpleBlock(32, 32)
        )
        self.upsample4 = BasicConvTransposeBNS3d(32, 32, kernel_size=2, stride=2)
        self.upBlock4 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1), InPlaceABNSync(32)
        )
        self.drop = nn.Dropout3d(p=0.5)
        self.FCN_out = nn.Conv3d(32, 3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x0 = self.downBlock1(x)
        x1 = self.pool1(x0)
        x1 = self.downBlock2(x1)
        x2 = self.pool2(x1)
        x2 = self.downBlock3(x2)
        x3 = self.pool3(x2)
        x3 = self.downBlock4(x3)
        x4 = self.pool4(x3)
        x4 = self.across(x4)
        x4 = self.upsample1(x4)
        x5 = self.upBlock1(x4 + x3)
        x5 = self.upsample2(x5)
        x6 = self.upBlock2(x5 + x2)
        x6 = self.upsample3(x6)
        x7 = self.upBlock3(x6 + x1)
        x7 = self.upsample4(x7)
        out = self.upBlock4(x7 + x0)
        out = self.drop(out)
        out = self.FCN_out(out)
        return self.softmax(out)


#############################################################################################################
############################################## Denseblock-Based模型 #########################################
class SingleLayer(nn.Module):
    def __init__(self, inChannels, growth_Channels, leaky_relu=False):
        super(SingleLayer, self).__init__()
        self.bn = nn.BatchNorm3d(inChannels)
        if leaky_relu:
            self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = torch.cat((x, out), 1)
        return out


class SingleLayerBN(nn.Module):
    def __init__(self, inChannels, growth_Channels):
        super(SingleLayerBN, self).__init__()
        self.bn = InPlaceABN(inChannels)
        self.conv = nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = torch.cat((x, out), 1)
        return out


class SingleLayerBNS(nn.Module):
    def __init__(self, inChannels, growth_Channels):
        super(SingleLayerBNS, self).__init__()
        self.bn = InPlaceABNSync(inChannels)
        self.conv = nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(self.bn(x))
        out = torch.cat((x, out), 1)
        return out


class Dense_Vnet1(nn.Module):
    def __init__(self):
        super(Dense_Vnet1, self).__init__()
        self.downblock1 = DownBlockBNS(1, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=4
        )
        self.downblock2 = DownBlockBNS(128, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock2 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=8
        )
        self.downblock3 = DownBlockBNS(256, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock3 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=12
        )

        self.skipconv1 = nn.Conv3d(128, 24, kernel_size=3, padding=1)
        self.skipconv2 = nn.Conv3d(256, 24, kernel_size=3, padding=1)
        self.skipconv3 = nn.Conv3d(384, 24, kernel_size=3, padding=1)

        self.FCN_out = PreConvBNS3d(72, 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(1)

    def _make_dense(self, inChannels, growth_Channels, nDenseLayers, leaky_relu=False):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayer(inChannels, growth_Channels, leaky_relu=leaky_relu)
                )
                inChannels += growth_Channels
        layers.append(nn.BatchNorm3d(inChannels))
        if leaky_relu:
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

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
        layers.append(InPlaceABN(inChannels))
        return nn.Sequential(*layers)

    def _make_denseBNS(self, inChannels, growth_Channels, nDenseLayers):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(SingleLayerBNS(inChannels, growth_Channels))
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
        return self.softmax(out)


class Dense_Vnet2(nn.Module):
    def __init__(self):
        super(Dense_Vnet2, self).__init__()
        self.downblock1 = DownBlockBNS(1, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=4
        )
        self.downblock2 = DownBlockBNS(128, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock2 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=8
        )
        self.downblock3 = DownBlockBNS(256, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock3 = self._make_denseBNS(
            inChannels=32, growth_Channels=32, nDenseLayers=12
        )

        self.skipconv1 = nn.Conv3d(128, 24, kernel_size=3, padding=1)
        self.skipconv2 = nn.Conv3d(256, 24, kernel_size=3, padding=1)
        self.skipconv3 = nn.Conv3d(384, 24, kernel_size=3, padding=1)

        self.FCN_out = PreConvBNS3d(72, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(1)

    def _make_dense(self, inChannels, growth_Channels, nDenseLayers, leaky_relu=False):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayer(inChannels, growth_Channels, leaky_relu=leaky_relu)
                )
                inChannels += growth_Channels
        layers.append(nn.BatchNorm3d(inChannels))
        if leaky_relu:
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

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
        layers.append(InPlaceABN(inChannels))
        return nn.Sequential(*layers)

    def _make_denseBNS(self, inChannels, growth_Channels, nDenseLayers):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(SingleLayerBNS(inChannels, growth_Channels))
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


class Dense_Unet1(nn.Module):
    def __init__(self):
        super(Dense_Unet1, self).__init__()
        self.downblock1 = DownBlockBNS(1, 32, kernel_size=3, stride=2, padding=1)
        self.denseblock1 = self._make_denseBNS(
            inChannels=32, growth_Channels=16, nDenseLayers=6
        )
        self.downblock2 = DownBlockBNS(96, 48, kernel_size=3, stride=2, padding=1)
        self.denseblock2 = self._make_denseBNS(
            inChannels=48, growth_Channels=16, nDenseLayers=12
        )
        self.downblock3 = DownBlockBNS(192, 96, kernel_size=3, stride=2, padding=1)
        self.denseblock3 = self._make_denseBNS(
            inChannels=96, growth_Channels=16, nDenseLayers=18
        )
        self.downblock4 = DownBlockBNS(288, 144, kernel_size=3, stride=2, padding=1)
        self.denseblock4 = self._make_denseBNS(
            inChannels=144, growth_Channels=16, nDenseLayers=24
        )
        self.upblock1 = BasicConvTransposeBNS3d(384, 192, kernel_size=2, stride=2)
        self.denseblock5 = self._make_denseBNS(
            inChannels=480, growth_Channels=16, nDenseLayers=18
        )
        self.upblock2 = BasicConvTransposeBNS3d(288, 144, kernel_size=2, stride=2)
        self.denseblock6 = self._make_denseBNS(
            inChannels=336, growth_Channels=16, nDenseLayers=12
        )
        self.upblock3 = BasicConvTransposeBNS3d(192, 96, kernel_size=2, stride=2)
        self.denseblock7 = self._make_denseBNS(
            inChannels=192, growth_Channels=16, nDenseLayers=6
        )
        self.upblock4 = BasicConvTransposeBNS3d(96, 16, kernel_size=2, stride=2)
        self.FCN_out = BasicConvBNS3d(16, 3, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(1)

    def _make_dense(self, inChannels, growth_Channels, nDenseLayers, leaky_relu=False):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(
                    SingleLayer(inChannels, growth_Channels, leaky_relu=leaky_relu)
                )
                inChannels += growth_Channels
        layers.append(nn.BatchNorm3d(inChannels))
        if leaky_relu:
            layers.append(nn.LeakyReLU(negative_slope=0.01, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_denseBN(self, inChannels, growth_Channels, nDenseLayers):
        layers = []
        for i in range(int(nDenseLayers)):
            if i == 0:
                layers.append(
                    nn.Conv3d(inChannels, growth_Channels, kernel_size=3, padding=1)
                )
                inChannels = growth_Channels
            else:
                layers.append(SingleLayerABN(inChannels, growth_Channels))
                inChannels += growth_Channels
        layers.append(InPlaceABN(inChannels))
        return nn.Sequential(*layers)

    def _make_denseBNS(self, inChannels, growth_Channels, nDenseLayers):
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
        x3 = self.downblock4(x2)
        x3 = self.denseblock4(x3)
        x3 = self.upblock1(x3)
        x4 = self.denseblock5(torch.cat((x2, x3), 1))
        x4 = self.upblock2(x4)
        x5 = self.denseblock6(torch.cat((x1, x4), 1))
        x5 = self.upblock3(x5)
        x6 = self.denseblock7(torch.cat((x, x5), 1))
        x6 = self.upblock4(x6)
        out = self.FCN_out(x6)
        return self.softmax(out)


#######################################################################################################
########################################### CUMED模型 #################################################


class MyResblock(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(MyResblock, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv1 = nn.Conv3d(inchannel, inchannel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(inchannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inchannel, outchannel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(outchannel)
        self.conv_res = nn.Conv3d(inchannel, outchannel, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.inchannel != self.outchannel:
            residual = self.conv_res(residual)
        out = out + residual
        out = self.relu(out)
        return out


class CUMED_1t3(nn.Module):
    def __init__(self):
        super(CUMED_1t3, self).__init__()
        self.preblock = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.downblock1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.resblock1 = MyResblock(32, 64)
        self.downblock2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.resblock2 = MyResblock(64, 128)
        self.downblock3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.resblock3 = MyResblock(128, 256)
        self.upblock1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        )
        self.resblock4 = MyResblock(128, 128)
        self.upblock2 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        )
        self.resblock5 = MyResblock(64, 64)
        self.upblock3 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        )
        self.resblock6 = MyResblock(32, 32)
        self.FCN_out = nn.Conv3d(32, 3, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        x1 = self.preblock(x)
        x2 = self.downblock1(x1)
        x2 = self.resblock1(x2)
        x3 = self.downblock2(x2)
        x3 = self.resblock2(x3)
        x4 = self.downblock3(x3)
        x4 = self.resblock3(x4)
        x4 = self.upblock1(x4)
        x5 = self.resblock4(x4 + x3)
        x5 = self.upblock2(x5)
        x6 = self.resblock5(x5 + x2)
        x6 = self.upblock3(x6)
        out = self.resblock6(x6 + x1)
        out = self.FCN_out(out)
        return self.softmax(out)
