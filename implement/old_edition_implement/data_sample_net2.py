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


def save_params(save_folder, name, rangez, factor):
    fp = open(save_folder + "/" + name + "_transparams.txt", "w")
    fp.write(str(rangez[0]) + "\n" + str(rangez[1]) + "\n")
    fp.write(str(factor[0]) + "\n" + str(factor[1]) + "\n" + str(factor[2]) + "\n")
    fp.close()
    return


def read_params(save_path):
    f = open(save_path, "r")
    lines = f.readlines()
    lowlim = int(lines[0])
    highlim = int(lines[1])
    factor1 = float(lines[2])
    factor2 = float(lines[3])
    factor3 = float(lines[4])
    return lowlim, highlim, factor1, factor2, factor3


def sample_data_for_net2(
    data256_path,
    save_path,
    size,
    padding=False,
    mode="constant",
    with_denoise=False,
    test=False,
):
    ######## data256_path: the folder containing raw & denoised datas with size of 256*256 ###############
    ######## save_path: a list with 2 items, corresponding to size of 128 and 144 in (x,y) ###############
    ######## padding and mode: switch of padding for shift argment #######################################
    filelist = os.listdir(data256_path)
    raw_datas = [
        os.path.join(data256_path, file) for file in filelist if "_raw_256.npy" in file
    ]
    if test:
        labelMs = [
            os.path.join(data256_path, file)
            for file in filelist
            if "_AD256.npy" in file
        ]
    else:
        labelMs = [
            os.path.join(data256_path, file) for file in filelist if "_M256.npy" in file
        ]
    raw_datas.sort()
    labelMs.sort()
    if with_denoise:
        dns_datas = [
            os.path.join(data256_path, file)
            for file in filelist
            if "_denoised_256.npy" in file
        ]
        dns_datas.sort()
        for raw_data, dns_data, labelM in zip(raw_datas, dns_datas, labelMs):
            start = time.time()
            num = raw_data.split("/")[-1].split("_")[0]
            labelm = np.load(labelM)
            M_z, _, _ = np.where(labelm)
            lowlim, highlim = M_z.min(), M_z.max() + 1  ###### 在训练过程中，根据labelm确定主动脉上下界
            raw = np.load(raw_data)
            raw_resized, factor = resize(raw[lowlim:highlim], size, order=2)
            np.save(os.path.join(save_path[0], num + "_raw_128.npy"), raw_resized)
            time_elapsed = time.time() - start
            print(
                "Raw data sample2 run {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

            start = time.time()
            dns = np.load(dns_data)
            dns_resized, _ = resize(dns[lowlim:highlim], size, order=2)
            np.save(os.path.join(save_path[0], num + "_denoised_128.npy"), dns_resized)
            save_params(save_path[0], num, [lowlim, highlim], factor)
            time_elapsed = time.time() - start
            print(
                "Raw data sample2 run {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )

            print(num + " raw", raw_resized.shape, raw_resized.dtype)
            print(num + " dns", dns_resized.shape, dns_resized.dtype)
            if padding:
                raw_resized_pad = np.pad(raw_resized, ((0, 0), (8, 8), (8, 8)), mode)
                dns_resized_pad = np.pad(dns_resized, ((0, 0), (8, 8), (8, 8)), mode)
                np.save(
                    os.path.join(save_path[1], num + "_raw_144.npy"), raw_resized_pad
                )
                np.save(
                    os.path.join(save_path[1], num + "_denoised_144.npy"),
                    dns_resized_pad,
                )
                print(num + " raw", raw_resized_pad.shape, raw_resized_pad.dtype)
                print(num + " dns", dns_resized_pad.shape, dns_resized_pad.dtype)
    else:
        for raw_data, labelM in zip(raw_datas, labelMs):
            start = time.time()
            num = raw_data.split("/")[-1].split("_")[0]
            raw = np.load(raw_data)
            labelm = np.load(labelM)
            M_z, _, _ = np.where(labelm)
            lowlim, highlim = M_z.min(), M_z.max() + 1  ###### 在训练过程中，根据labelm确定主动脉上下界
            raw_resized, factor = resize(raw[lowlim:highlim], size, order=2)
            save_params(save_path[0], num, [lowlim, highlim], factor)
            np.save(os.path.join(save_path[0], num + "_raw_128.npy"), raw_resized)
            print(num + " raw", raw_resized.shape, raw_resized.dtype)
            time_elapsed = time.time() - start
            print(
                "Data sample2 run {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
            if padding:
                raw_resized_pad = np.pad(raw_resized, ((0, 0), (8, 8), (8, 8)), mode)
                np.save(
                    os.path.join(save_path[1], num + "_raw_144.npy"), raw_resized_pad
                )
                print(num + " raw", raw_resized_pad.shape, raw_resized_pad.dtype)


def sample_label_for_net2(
    data256_path, save_path, size, padding=False, mode="constant"
):
    ######## data256_path: the folder containing raw & denoised datas with size of 256*256 ###############
    ######## save_path: a list with 2 items, corresponding to size of 128 and 144 in (x,y) ###############
    ######## padding and mode: switch of padding for shift argment                         ###############
    filelist = os.listdir(data256_path)
    labelMs = [
        os.path.join(data256_path, file) for file in filelist if "_M256.npy" in file
    ]
    labelTs = [
        os.path.join(data256_path, file) for file in filelist if "_T256.npy" in file
    ]
    labelFs = [
        os.path.join(data256_path, file) for file in filelist if "_F256.npy" in file
    ]
    labelMs.sort()
    labelTs.sort()
    labelFs.sort()
    for labelM, labelT, labelF in zip(labelMs, labelTs, labelFs):
        num = labelM.split("/")[-1].split("_")[0]
        labelm = np.load(labelM)
        labelt = np.load(labelT)
        labelf = np.load(labelF)
        M_z, _, _ = np.where(labelm)
        lowlim, highlim = M_z.min(), M_z.max() + 1  ###### 在训练过程中，根据labelm确定主动脉上下界
        labelt_resized, factor = resize(labelt[lowlim:highlim], size, order=1)
        labelf_resized, _ = resize(labelf[lowlim:highlim], size, order=1)
        labelt_resized[labelt_resized > 0] = 1
        labelt_resized = labelt_resized.astype(np.uint8)
        labelf_resized[labelf_resized > 0] = 1
        labelf_resized = labelf_resized.astype(np.uint8)
        labeltf_resized = labelf_resized.copy() * 2
        labeltf_resized[labelt_resized > 0] = 1
        np.save(os.path.join(save_path[0], num + "_T128.npy"), labelt_resized)
        np.save(os.path.join(save_path[0], num + "_F128.npy"), labelf_resized)
        np.save(os.path.join(save_path[0], num + "_TF128.npy"), labeltf_resized)
        print(num, labeltf_resized.shape, labeltf_resized.dtype)
        if padding:
            labelt_resized_pad = np.pad(labelt_resized, ((0, 0), (8, 8), (8, 8)), mode)
            labelf_resized_pad = np.pad(labelf_resized, ((0, 0), (8, 8), (8, 8)), mode)
            labeltf_resized_pad = np.pad(
                labeltf_resized, ((0, 0), (8, 8), (8, 8)), mode
            )
            np.save(os.path.join(save_path[1], num + "_T144.npy"), labelt_resized_pad)
            np.save(os.path.join(save_path[1], num + "_F144.npy"), labelf_resized_pad)
            np.save(os.path.join(save_path[1], num + "_TF144.npy"), labeltf_resized_pad)
            print(num, labeltf_resized_pad.shape, labeltf_resized_pad.dtype)


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
