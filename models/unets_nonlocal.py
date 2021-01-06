import torch
import torch.nn.functional as F
import torch.nn as nn
from .inplace_abn import InPlaceABN, InPlaceABNSync, ABN
from .base_nets.layer import Bottleneck_inplace

class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, abn, stride=1, kernel_size=3, downsample=None,):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(inplanes)
        # self.bn1 = abn(inplanes)
        # self.relu = nn.ReLU6(inplace=True)
        self.relu = nn.ReLU6()
        self.conv1 = nn.Conv3d(inplanes, planes, stride=stride, kernel_size=kernel_size, padding=int((kernel_size - 1)/2))
        
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2 = abn(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, padding=int((kernel_size - 1)/2))
        
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        
        out = x.clone()

        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        
        # 保持正确的梯度流动：x -> out -> x + out 
        # out += x 
        out = out + x

        return out

class ResidualConnect(nn.Module):

    def __init__(self, md, inplanes, planes, res_type='SAME'):
        super(ResidualConnect, self).__init__()
        """
        :param md -> nn.Module: 模块
        :param res_type -> str: ['SAME', 'UP', 'DOWN']
        
        """
        self.md = md
        self.res_type = res_type
        self.downsample = nn.Conv3d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.upsample = nn.ConvTranspose3d(inplanes, planes, kernel_size=3, stride=2, padding=1, output_padding=1)


    def forward(self, x):

        out = x.clone()

        out = self.md(out)
        if self.res_type == 'UP':
            x = self.upsample(x)
        elif self.res_type == 'DOWN':
            x = self.downsample(x)
        
        out = out + x

        return out



class MultiHeadAttention(torch.nn.Module):
    def __init__(self, in_channel, key_filters, value_filters,
                 output_filters, num_heads, dropout_prob=0.5, layer_type='SAME'):
        super().__init__()
        """
        Multihead scaled-dot-product attention (3d) with input/output transformations.

        :param inputs -> tensor: [batch, c, d, h, w]
        :param in_channel -> int: 输入通道数量
        :param key_filters -> int: k-transform后的通道数
        :param value_filters -> int: v-transform后的通道数
        :param output_filters -> int: 输出通道数
        :param num_heads -> int: 需要被key_filters & value_filters整除
        :param layer_type -> str: choose from ['SAME', 'DOWN', 'UP']

        Raises:
            ValueError: attention heads的数量不能被通道数整除.
        """

        if key_filters % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (key_filters, num_heads))
        if value_filters % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (value_filters, num_heads))
        if layer_type not in ['SAME', 'DOWN', 'UP']:
            raise ValueError("Layer type (%s) must be one of SAME, "
                             "DOWN, UP." % (layer_type))

        self.num_heads = num_heads
        self.layer_type = layer_type

        self.QueryTransform = None
        if layer_type == 'SAME':
            self.QueryTransform = nn.Conv3d(in_channel, key_filters, kernel_size=1, stride=1,
                                            padding=0, bias=True)
        elif layer_type == 'DOWN':
            self.QueryTransform = nn.Conv3d(in_channel, key_filters, kernel_size=3, stride=2,
                                            padding=1, bias=True)  # author use bias
        elif layer_type == 'UP':
            self.QueryTransform = nn.ConvTranspose3d(in_channel, key_filters, kernel_size=3, stride=2,
                                                     padding=1, bias=True)

        self.KeyTransform = nn.Conv3d(in_channel, key_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.ValueTransform = nn.Conv3d(in_channel, value_filters, kernel_size=1, stride=1, padding=0, bias=True)
        self.attention_dropout = nn.Dropout(dropout_prob)

        self.outputConv = nn.Conv3d(value_filters, output_filters, kernel_size=1, stride=1, padding=0, bias=True)

        self._scale = (key_filters // num_heads) ** 0.5

    def forward(self, inputs):
        """
        :param inputs: B, C, D, H, W
        :return: outputs: B, Co, Dq, Hq, Wq
        """

        if self.layer_type == 'SAME' or self.layer_type == 'DOWN':
            q = self.QueryTransform(inputs)
        elif self.layer_type == 'UP':
            q = self.QueryTransform(inputs, output_size=(inputs.shape[2] * 2, inputs.shape[3] * 2, inputs.shape[4] * 2))

        # [B, Dq, Hq, Wq, Ck]
        k = self.KeyTransform(inputs).permute(0, 2, 3, 4, 1)
        v = self.ValueTransform(inputs).permute(0, 2, 3, 4, 1)
        q = q.permute(0, 2, 3, 4, 1)

        Batch, Dq, Hq, Wq = q.shape[0], q.shape[1], q.shape[2], q.shape[3]

        # [B, D, H, W, N, Ck]
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)
        q = self.split_heads(q, self.num_heads)

        # [(B, D, H, W, N), c]
        k = torch.flatten(k, 0, 4)
        v = torch.flatten(v, 0, 4)
        q = torch.flatten(q, 0, 4)

        # normalize
        q = q / self._scale
        # attention
        # [(B, Dq, Hq, Wq, N), (B, D, H, W, N)]
        A = torch.matmul(q, k.transpose(0, 1))
        A = torch.softmax(A, dim=1)
        A = self.attention_dropout(A)

        # [(B, Dq, Hq, Wq, N), C]
        O = torch.matmul(A, v)
        # [B, Dq, Hq, Wq, C]
        O = O.view(Batch, Dq,Hq, Wq, v.shape[-1] * self.num_heads)
        # [B, C, Dq, Hq, Wq]
        O = O.permute(0, 4, 1, 2, 3)
        # [B, Co, Dq, Hq, Wq]
        O = self.outputConv(O)

        return O

    def split_heads(self, x, num_heads):
        """
        把通道split成为若干head

        :param x -> tensor: [batch, h, w, channels]
        :param num_heads: head数量
        :return
            a Tensor with shape [batch, h, w, num_heads, channels / num_heads]
        """

        channel_num = x.shape[-1]
        return x.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], num_heads, int(channel_num / num_heads))



class multiClass_unet_nonLocal(nn.Module):
    def __init__(self, n_inp=1, n_out=3, feats=(32, 64, 128), abn=2, n_encoders=2,):
        """
        :param n_inp: 输入channel数量，3D图像默认为1
        :param n_out: 输出channel数量，与n_inp一致
        :param feats: 经过conv后的channel数量
        :param abn: [0,1,2]指定类型的abn
        :param n_encoders:

        """
        super().__init__()
        self.n_inp = n_inp
        if abn == 0:
            abnblock = ABN
        elif abn == 1:
            abnblock = InPlaceABN
        elif abn == 2:
            abnblock = InPlaceABNSync
        # abnblock = abn

        self.relu = nn.ReLU6(inplace=True)

        self.in_layer = ResidualBlock(inplanes=n_inp, planes=feats[0], kernel_size=3, abn=abnblock,
                                      downsample=nn.Conv3d(n_inp, feats[0], stride=1, kernel_size=1))
        self.down_layer1 = ResidualBlock(inplanes=feats[0], planes=feats[1], abn=abnblock, stride=2,
                                         downsample=nn.Conv3d(feats[0], feats[1], stride=2, kernel_size=1))
        self.down_layer2 = ResidualBlock(inplanes=feats[1], planes=feats[2], abn=abnblock, stride=2,
                                         downsample=nn.Conv3d(feats[1], feats[2], stride=2, kernel_size=1))
        
        # github:https://github.com/divelab/Non-local-U-Nets
        # 源码, 三类channel数量一致, head2 = 2
        self.bottom_layer = ResidualConnect(MultiHeadAttention(in_channel=feats[2], 
                                                               key_filters=feats[2],
                                                               value_filters=feats[2],
                                                               output_filters=feats[2],
                                                               num_heads=2,
                                                               layer_type='SAME'),
                                            inplanes=feats[2],
                                            planes=feats[2],
                                            res_type='SAME',)
        
        # heads = 1
        self.up_layer2 = ResidualConnect(MultiHeadAttention(in_channel=feats[2], 
                                                            key_filters=feats[1],
                                                            value_filters=feats[1],
                                                            output_filters=feats[1],
                                                            num_heads=1,
                                                            layer_type='UP'),
                                        inplanes=feats[2],
                                        planes=feats[1],
                                        res_type='UP',)
                              

        self.up_layer1 = ResidualConnect(MultiHeadAttention(in_channel=feats[1], 
                                                            key_filters=feats[0],
                                                            value_filters=feats[0],
                                                            output_filters=feats[0],
                                                            num_heads=1,
                                                            layer_type='UP'),
                                        inplanes=feats[1],
                                        planes=feats[0],
                                        res_type='UP',
                                        )
        
        self.out_layer = ResidualBlock(inplanes=feats[0], planes=n_out, kernel_size=1, abn=abnblock,
                                       downsample=nn.Conv3d(feats[0], n_out, stride=1, kernel_size=1))


    def forward(self, x, *args):

        x0 = self.in_layer(x)
        x1 = self.down_layer1(x0)
        x2 = self.down_layer2(x1)
        # print(x2.shape)
        
        xb = self.bottom_layer(x2)
        # print(xb.shape)

        xr2 = self.up_layer2(xb)
        xr21 = torch.add(xr2, x1)
        xr1 = self.up_layer1(xr2)
        xr11 = torch.add(xr1, x0)

        out = self.out_layer(xr11)

        return out


if __name__ == '__main__':
    device = torch.device('cpu')
    inputs = torch.rand(1, 24, 60, 60).unsqueeze(0).to(device)
    # net = MultiHeadAttention(in_channel=1, 
    #                          key_filters=10, 
    #                          value_filters=8, 
    #                          output_filters=5, 
    #                          num_heads=1, 
    #                          dropout_prob=0.5, 
    #                          layer_type='UP', # 'SAME', 'DOWN', 'UP'
    #                          ) 
    # net = nn.ConvTranspose3d(in_channels=1, out_channels=10, kernel_size=3, stride=2, padding=1, output_padding=1)
    net = multiClass_unet_nonLocal(
        n_inp=1,
        n_out=3,
        feats=(32, 64, 128),
        abn=2,
    )
    res = net(inputs)
    print('input shape: {}'.format(inputs.shape))
    print('res shape: {}'.format(res.shape))