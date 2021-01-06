import torch
import numpy as np


def mypad(x, pad, padv):
    def neg(b):
        if b == 0:
            return None
        else:
            return -b

    pad = np.array(pad)
    x2 = np.zeros(np.array(x.shape) + np.sum(pad, 1), dtype=x.dtype) + padv
    x2[
        pad[0, 0] : neg(pad[0, 1]),
        pad[1, 0] : neg(pad[1, 1]),
        pad[2, 0] : neg(pad[2, 1]),
        pad[3, 0] : neg(pad[3, 1]),
    ] = x
    return x2


class SplitComb:
    def __init__(self, config):
        if "margin_inference" in config["prepare"]:
            self.margin = self.init(
                config["prepare"]["margin_inference"]
            )  ### 都会引入margin这个参数，虚拟的边界扩充
        else:
            self.margin = self.init(config["prepare"]["margin"])
        self.side_len = np.array(config["prepare"]["crop_size"]) - self.margin * 2
        self.stride = self.init(config["prepare"]["seg_stride"])  ### 现在还不太懂这个stride是啥
        self.pad_mode = config["prepare"]["pad_mode"]
        self.pad_value = config["prepare"]["pad_value"]

    @staticmethod
    def init(x):
        if isinstance(x, int):
            return np.array([x, x, x])
        else:
            return np.array(x)

    @staticmethod
    def getse(izhw, nzhw, crop_size, side_len, shape_post):
        se = []
        for i, n, crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_post):
            if i == n - 1 and i > 0:
                e = shape  ### 如果是最后一个，需要保证crop的大小一致，可以从后往前查crop的大小
                s = e - crop
            else:
                s = i * side
                e = s + crop
            se += [s, e]
        return se

    @staticmethod
    def getse2(izhw, nzhw, crop_size, side_len, shape_len):
        se = []
        for i, n, crop, side, shape in zip(izhw, nzhw, crop_size, side_len, shape_len):
            if i == n - 1 and i > 0:
                e = shape
                s = e - side
            else:
                s = i * side
                e = s + side
            se += [s, e]
        return se

    def split(self, data, side_len=None, margin=None):
        if side_len is None:
            side_len = self.side_len
        if margin is None:
            margin = self.margin
        crop_size = side_len + margin * 2

        assert np.all(side_len > margin)

        splits = []
        _, z, h, w = data.shape

        nz = int(
            np.ceil(float(z) / side_len[0])
        )  ### 这个地方除以的是side_len，是pad之前的大小进行分块，可以分成多少个有效区间
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        shape_pre = [z, h, w]

        pad = [
            [0, 0],
            [margin[0], np.max([margin[0], crop_size[0] - z - margin[0]])],
            [margin[1], np.max([margin[1], crop_size[1] - h - margin[1]])],
            [margin[2], np.max([margin[2], crop_size[2] - w - margin[2]])],
        ]  ### 测试margin也是虚拟的，但必须保证在pad的大小和margin一样
        #         print(data.shape)
        #         print(side_len[1])
        #         print(side_len[1]-h-margin[1])
        #         print(pad)
        if self.pad_mode == "constant":
            data = mypad(data, pad, self.pad_value)
        else:
            data = np.pad(data, pad, self.pad_mode)
        shape_post = list(data.shape[1:])
        shapes = np.array([shape_pre, shape_post])
        self.shapes = shapes
        #         split_data = Data_bysplit(data, [nz,nh,nw], crop_size, side_len, shape_post)
        #         splits = np.zeros((nz*nh*nw, data.shape[1], crop_size[0], crop_size[1], crop_size[2]), dtype = data.dtype)
        splits = []
        id = 0
        for iz in range(nz):  ### z轴有n个
            for ih in range(nh):  ### y轴有1个
                for iw in range(nw):  ### x轴有1个
                    sz, ez, sh, eh, sw, ew = self.getse(
                        [iz, ih, iw], [nz, nh, nw], crop_size, side_len, shape_post
                    )  ### 这样的话，取得不是中间
                    #                     print(sz, ez, sh, eh, sw, ew)
                    splits.append(data[:, sz:ez, sh:eh, sw:ew])
                    id += 1
        splits = np.array(splits)  ### 为啥要套上个（）,感觉他们之间都是有重叠的。。。。
        return splits, shapes

    def combine(self, output, shapes=None, side_len=None, stride=None, margin=None):

        if side_len is None:
            side_len = self.side_len
        if stride is None:
            stride = self.stride
        if margin is None:
            margin = self.margin
        if shapes is None:
            shape = self.shapes

        shape_pre, shape_post = shapes

        z, h, w = shape_pre  #### margin之前的长宽高
        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        assert np.all(side_len % stride == 0)
        assert np.all(margin % stride == 0)

        newshape = (np.array([z, h, w]) / stride).astype(np.int)
        side_len = (self.side_len / stride).astype(np.int)
        margin = (self.margin / stride).astype(np.int)
        crop_size = side_len + margin * 2

        splits = []
        for i in range(len(output)):  #### output是一个batch里的几个tensor
            splits.append(output[i])  ### 拆解，把batch维度去掉

        if isinstance(output[0], torch.Tensor):
            occur = torch.zeros(
                (1, nz * side_len[0], nh * side_len[1], nw * side_len[2]),
                dtype=output[0].dtype,
                device=output[0].device,
            )
            output = torch.zeros(
                (
                    splits[0].shape[0],  ### 信道维度保留
                    nz * side_len[0],
                    nh * side_len[1],
                    nw * side_len[2],
                ),
                dtype=output[0].dtype,
                device=output[0].device,
            )
        else:
            occur = np.zeros(
                (1, nz * side_len[0], nh * side_len[1], nw * side_len[2]),
                output[0].dtype,
            )
            output = -1000000 * np.ones(
                (
                    splits[0].shape[0],
                    nz * side_len[0],
                    nh * side_len[1],
                    nw * side_len[2],
                ),
                output[0].dtype,
            )
        #         print(output.shape)
        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz, ez, sh, eh, sw, ew = self.getse2(
                        [iz, ih, iw], [nz, nh, nw], crop_size, side_len, shape_pre
                    )  ### 往回填充用的preshape
                    #                     print(sz, ez, sh, eh, sw, ew)
                    sz, ez, sh, eh, sw, ew = int(sz), int(ez), int(sh), int(eh), int(sw), int(ew)
                    split = splits[idx][
                        :,
                        margin[0] : margin[0] + side_len[0],
                        margin[1] : margin[1] + side_len[1],
                        margin[2] : margin[2] + side_len[2],
                    ]
                    output[
                        :, sz:ez, sh:eh, sw:ew
                    ] += split  ### 所以这个东西，在有margin的时候，比cropsize小一圈
                    occur[:, sz:ez, sh:eh, sw:ew] += 1
                    idx += 1
        #### 这是取出和原图一大小的区域 ####
        return (
            output[:, : newshape[0], : newshape[1], : newshape[2]]
            / occur[:, : newshape[0], : newshape[1], : newshape[2]]
        )
