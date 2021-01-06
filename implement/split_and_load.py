import os
import time
from torch.utils.data import Dataset
from scipy.ndimage import zoom, rotate, binary_dilation
import warnings
import numpy as np


class Split:
    def __init__(self, config):
        margin = np.array(config.prepare["margin"])
        side_len = np.array(config.prepare["crop_size"]) - margin * 2
        stride = config.prepare["seg_stride"]
        pad_value = config.prepare["pad_value"]

        if isinstance(side_len, int):
            side_len = [side_len] * 3
        if isinstance(stride, int):
            stride = [stride] * 3
        if isinstance(margin, int):
            margin = [margin] * 3

        self.side_len = np.array(side_len)
        self.stride = np.array(stride)
        self.margin = np.array(margin)
        self.pad_value = pad_value

    def split(self, data, side_len=None, margin=None, mode="constant"):
        if side_len is None:
            side_len = self.side_len
        if margin is None:
            margin = self.margin

        assert np.all(side_len > margin)

        splits = []
        _, z, h, w = data.shape

        nz = int(np.ceil(float(z) / side_len[0]))
        nh = int(np.ceil(float(h) / side_len[1]))
        nw = int(np.ceil(float(w) / side_len[2]))

        zhw = [z, h, w]
        self.zhw = zhw

        pad = [
            [0, 0],
            [margin[0], nz * side_len[0] - z + margin[0]],
            [margin[1], nh * side_len[1] - h + margin[1]],
            [margin[2], nw * side_len[2] - w + margin[2]],
        ]

        if mode == "constant":
            data = np.pad(data, pad, "constant", constant_values=self.pad_value)
        else:
            data = np.pad(data, pad, mode=mode)

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len[0]
                    ez = (iz + 1) * side_len[0] + 2 * margin[0]
                    sh = ih * side_len[1]
                    eh = (ih + 1) * side_len[1] + 2 * margin[1]
                    sw = iw * side_len[2]
                    ew = (iw + 1) * side_len[2] + 2 * margin[2]

                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)

        splits = np.concatenate(splits, 0)
        return splits, zhw


class myDataset(Dataset):
    def __init__(self, config):

        self.config = config
        self.split = Split(config)
        self.data_path = config.prepare["data_dir"]
        self.img_files, self.cases = get_all_dicompaths(self.data_path)

    def __getitem__(self, idx, debug=False):

        idx = idx % len(self.cases)
        img_raw = np.load(self.img_files[idx])  ### 根据名字加载npy，这个是已经预处理之后的数据
        if len(img_raw.shape) == 3:
            img_raw = img_raw[np.newaxis]
        img = img_raw[
            :,
            :,
            int((img_raw.shape[2] - self.config.prepare["crop_size"][1]) / 2) : int(
                (img_raw.shape[2] + self.config.prepare["crop_size"][1]) / 2
            ),
            int((img_raw.shape[3] - self.config.prepare["crop_size"][2]) / 2) : int(
                (img_raw.shape[3] + self.config.prepare["crop_size"][2]) / 2
            ),
        ]
        crop_img, nswh = self.split.split(img)
        crop_img = crop_img.astype("float32")

        return crop_img, nswh, self.cases[idx]

    def __len__(self):
        if self.config.debug:
            return 4
        else:
            if self.phase == "train":
                return len(self.cases) * self.config.train["train_repeat"]
            else:
                return len(self.cases)
