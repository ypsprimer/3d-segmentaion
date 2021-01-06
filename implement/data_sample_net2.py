import numpy as np
import os
import time
from scipy.ndimage.interpolation import zoom


def resize(
    imgs, size=None, resize_factor=None, backsize=False, mode="nearest", order=2
):
    if backsize:
        if np.all(resize_factor) == None:
            raise AttributeError
        else:
            back_factor = 1 / resize_factor
            imgback = zoom(imgs, back_factor, mode=mode, order=order)
            return imgback
    else:
        if len(imgs.shape) == 3:
            xy_factor = size[2] / imgs.shape[2]
            z_factor = size[0] / imgs.shape[0]
            resize_factor = np.array(
                [float(z_factor), float(xy_factor), float(xy_factor)]
            )
            newimgs = zoom(imgs, resize_factor, mode=mode, order=order)
            return newimgs, resize_factor
        else:
            raise ValueError("wrong shape")


if __name__ == "__main__":
    ######### 获取全部待处理的imgs的路径 ############
    data_path = "/ssd/data_256"  ##### saving datas with size of 256*256
    save_path = [
        "/ssd/data_128",
        "/ssd/data_144",
    ]  ##### saving datas with size of 128*128/144*144
    size = np.array([384, 128, 128])
    sample_data_for_net2(data_path, save_path, size, with_denoise=True)
    sample_label_for_net2(data_path, save_path, size)
