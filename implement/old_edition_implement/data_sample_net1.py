import numpy as np
import os, sys
import matplotlib.pyplot as plt
import time
import vtk
import torch
import ipyvolume
import pydicom
import SimpleITK as sitk
import matplotlib.image as mpimg
import random
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_opening,
)
from scipy.ndimage.interpolation import zoom
from skimage import data, filters, feature, segmentation, measure, morphology
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma


def lumTrans(img, win_low, win_high):
    lungwin = np.array([win_low, win_high])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg


def shuffle_and_split(rawdata_path, split_sets=6):
    save_path = rawdata_path
    newfilelist = [
        f.split("_raw_256.npy")[0]
        for f in os.listdir(save_path)
        if f.endswith("_raw_256.npy")
    ]
    random.shuffle(newfilelist)
    fold_size = int(len(newfilelist) / split_sets)
    for i in range(split_sets):
        subset = newfilelist[i * fold_size : (i + 1) * fold_size]
        valset = newfilelist[(i + 1) * fold_size : (i + 2) * fold_size]
        recent_list = newfilelist.copy()
        for sub1, sub2 in zip(subset, valset):
            recent_list.remove(sub1)
            recent_list.remove(sub2)
        trainset = recent_list
        for case in subset:
            fp = open(save_path + "/" + "subset{}.txt".format(i + 1), "a")
            fp.write(case + "\n")
            fp.close
        for case in valset:
            fp = open(save_path + "/" + "valset{}.txt".format(i + 1), "a")
            fp.write(case + "\n")
            fp.close
        for case in trainset:
            fp = open(save_path + "/" + "trainset{}.txt".format(i + 1), "a")
            fp.write(case + "\n")
            fp.close


def load_itk_image(path, ID):
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(path)
    dicom_names = reader.GetGDCMSeriesFileNames(path, series_IDs[ID])
    reader.SetFileNames(dicom_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    spacing = image.GetSpacing()  # x, y, z
    numpySpacing = np.array(list(reversed(spacing)))
    PationtID = reader.GetMetaData(0, "0010|0020")
    return image_array, numpySpacing, PationtID


def save_VOI(
    save_folder, name, z1, y1, x1, z2, y2, x2, spacing_z, spacing_y, spacing_x
):
    fp = open(save_folder + "/" + name + "_VOI.txt", "w")
    fp.write(
        str(int(z1))
        + "\n"
        + str(int(y1))
        + "\n"
        + str(int(x1))
        + "\n"
        + str(int(z2))
        + "\n"
    )
    fp.write(str(int(y2)) + "\n" + str(int(x2)) + "\n")
    fp.write(str(spacing_z) + "\n" + str(spacing_y) + "\n" + str(spacing_x) + "\n")
    fp.close()
    return


def read_VOIrange_spacing(case_path):
    f = open(case_path, "r")
    lines = f.readlines()
    pt1_z = int(lines[0])
    pt1_y = int(lines[1])
    pt1_x = int(lines[2])
    pt2_z = int(lines[3])
    pt2_y = int(lines[4])
    pt2_x = int(lines[5])
    sp_z = float(lines[6])
    sp_y = float(lines[7])
    sp_x = float(lines[8])
    pt1 = [pt1_z, pt1_y, pt1_x]
    pt2 = [pt2_z, pt2_y, pt2_x]
    sp = [sp_z, sp_y, sp_x]
    return pt1, pt2, sp


def stl_writer(filename, stack, spacing, dcm_folder):
    ###### filename: the file name of the output ".stl" #######
    ###### stack: the volume of 3D ndarray ####################
    ###### spacing: resolution of raw data ####################
    DEBUG = False
    RElAXATIONFACTOR = 0.03
    ITERATIONS = 100
    stack = stack.transpose(0, 1, 2)
    stack[stack >= 0.5] = 1.0
    stack[stack <= 0.5] = 0.0
    stack = stack * 255
    stack = np.require(stack, dtype=np.uint8)
    data_string = stack.tostring()
    #### 转换成vtk-image的形式，调用vtkImageImport类 ############
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)  #### importer数据只有灰度信息，不是rgb之类的
    #### vtk uses an array in the order : height, depth, width which is different of numpy (w,h,d)
    w, d, h = stack.shape
    dataImporter.SetDataExtent(0, h - 1, 0, d - 1, 0, w - 1)
    dataImporter.SetDataSpacing(spacing[2], spacing[1], spacing[0])
    # dataImporter.SetWholeExtent(0, h-1, 0, d-1, 0, w-1)
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dcm_folder)
    reader.Update()
    dcmImagePosition = reader.GetImagePositionPatient()
    dataImporter.SetDataOrigin(dcmImagePosition)
    threshold = vtk.vtkImageThreshold()
    threshold.SetInputConnection(dataImporter.GetOutputPort())
    threshold.ThresholdByLower(128)
    threshold.ReplaceInOn()
    threshold.SetInValue(0)
    threshold.ReplaceOutOn()
    threshold.SetOutValue(1)
    threshold.Update()
    dmc = vtk.vtkDiscreteMarchingCubes()
    dmc.SetInputConnection(threshold.GetOutputPort())
    dmc.GenerateValues(1, 1, 1)
    dmc.Update()
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputConnection(dmc.GetOutputPort())
    smooth.SetRelaxationFactor(RElAXATIONFACTOR)
    smooth.SetNumberOfIterations(ITERATIONS)
    writer = vtk.vtkSTLWriter()
    writer.SetInputConnection(smooth.GetOutputPort())
    writer.SetFileTypeToBinary()
    writer.SetFileName(filename)
    writer.Write()
    print("complete")


def get_VOIrange(data_itk, crop_size, raw_spacing, new_spacing):
    z_range = int(data_itk.shape[0] / 2)
    data = data_itk[z_range:].copy()
    data[data <= 100] = 0  ### 初步定位血管的大致区域，目前是根据根据阈值特性[200Hu-500Hu]
    data[data >= 700] = 0
    data[data > 0] = 1
    data = binary_opening(data, iterations=12)
    label_data = measure.label(data, connectivity=1)
    regionlist = sorted(
        measure.regionprops(label_data), key=lambda x: x.area, reverse=True
    )
    z_min = np.min(regionlist[0].coords[:, 0])
    z_max = np.min(regionlist[0].coords[:, 0])
    y_min = np.min(regionlist[0].coords[:, 1])
    y_max = np.max(regionlist[0].coords[:, 1])
    x_min = np.min(regionlist[0].coords[:, 2])
    x_max = np.max(regionlist[0].coords[:, 2])
    if z_min < data.shape[0] / 2:
        y1 = y_min
        y2 = y_max + 40
        x1 = x_min - 40
        x2 = x_max
    else:
        y1 = min(y_min, np.min(regionlist[1].coords[:, 1]))
        y2 = max(y_max, np.max(regionlist[1].coords[:, 1])) + 40
        x1 = min(x_min, np.min(regionlist[1].coords[:, 2])) - 40
        x2 = max(x_max, np.max(regionlist[1].coords[:, 2]))
    y1 = max(y1, 0)
    y2 = min(y2, data.shape[1])
    x1 = max(x1, 0)
    x2 = min(x2, data.shape[2])
    real_size = int(np.round(crop_size * new_spacing[1] / raw_spacing[1]))
    if int(np.round((y2 + y1 - real_size) / 2)) < 0:
        y_start = 0
    elif int(np.round((y2 + y1 - real_size) / 2)) + real_size > data.shape[1]:
        y_start = data.shape[1] - real_size
    else:
        y_start = int(np.round((y2 + y1 - real_size) / 2))
    if int(np.round((x2 + x1 - real_size) / 2)) < 0:
        x_start = 0
    elif int(np.round((x2 + x1 - real_size) / 2)) + real_size > data.shape[2]:
        x_start = data.shape[2] - real_size
    else:
        x_start = int(np.round((x2 + x1 - real_size) / 2))
    y_min = y_start
    y_max = y_start + real_size
    x_min = x_start
    x_max = x_start + real_size
    z_min = 0
    z_max = data_itk.shape[0]
    return int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)


def VOI_resample(VOI, raw_spacing, new_spacing, crop_size, mode="nearest", order=2):
    if len(VOI.shape) == 3:
        new_z_axis = np.round(VOI.shape[0] * raw_spacing[0] / new_spacing[0])
        new_shape = [new_z_axis, float(crop_size), float(crop_size)]
        resize_factor = new_shape / np.array(VOI.shape)
        VOI_256 = zoom(VOI, resize_factor, mode=mode, order=order)
        return VOI_256
    else:
        raise ValueError("wrong shape")


def VOI_recover(output_256, saved_voi_range, mode="nearest", order=2):
    resize_factor = saved_voi_range / output_256.shape
    VOI = zoom(output_256, resize_factor, mode=mode, order=order)
    VOI[VOI > 0] = 1
    return VOI


def get_all_dicompaths(data_root):
    #### 输入dicom及label的存储目录 #####
    dicompath = data_root
    dicomdirs = [
        os.path.join(dicompath, num) for num in os.listdir(dicompath) if "i" not in num
    ]
    dicomdirs.sort()
    caselists = []
    for dicomdir in dicomdirs:
        datadir = [
            os.path.join(dicomdir, sub) for sub in os.listdir(dicomdir) if "com" in sub
        ][0]
        caselists.append(datadir)
    return caselists


def get_all_labelpaths(data_root):
    #### 输入dicom及label的存储目录 #####
    labelpath = data_root
    labeldirs = [
        os.path.join(labelpath, num) for num in os.listdir(labelpath) if "i" not in num
    ]
    labeldirs.sort()
    TLlists = []
    FLlists = []
    MLlists = []
    for labeldir in labeldirs:
        TLdir = [
            os.path.join(labeldir, sub) for sub in os.listdir(labeldir) if "TL" in sub
        ][0]
        TLlists.append(TLdir)
        FLdir = [
            os.path.join(labeldir, sub) for sub in os.listdir(labeldir) if "FL" in sub
        ][0]
        FLlists.append(FLdir)
        MLdir = [
            os.path.join(labeldir, sub) for sub in os.listdir(labeldir) if "mask" in sub
        ][0]
        MLlists.append(MLdir)
    return MLlists, TLlists, FLlists


def data_preprocessing_and_save(
    all_cases_dir, savepath, crop_size, new_spacing, denoise=False
):
    ########### caselists：输入提取出的dicom的目录列表 ########################################
    ########### savepath：存512*512的npy（包括raw和denoised）和256*256的crop坐标范围的目录 ####
    ########### crop_size：输入训练数据的(y,x)的归一化后尺寸 ##################################
    ########### new_spacing：输入训练数据的（z,y,x）的分辨率 ##################################
    ########### denoise：可选择是否需要降噪并储存降噪数据 #####################################
    caselists = get_all_dicompaths(
        all_cases_dir
    )  #### 数据存储路径：./0005/0005-pre-dicom/0000.dcm
    for index, case_path in enumerate(caselists):
        start = time.time()
        #### 从dicom子路径中读取数据，存储为原始数据raw512 #######
        num = case_path.split("/")[-2]
        data_raw, raw_spacing, PatientID = load_itk_image(case_path, 0)
        data_raw = data_raw.astype(np.int16)
        np.save(os.path.join(savepath[0], num + "_raw_512.npy"), data_raw)
        print(index, num, data_raw.shape, data_raw.dtype)

        #### 提取数据的VOI区域，数据截取并存储 #####
        z_min, y_min, x_min, z_max, y_max, x_max = get_VOIrange(
            data_raw, crop_size, raw_spacing, new_spacing
        )
        data_raw_crop = VOI_resample(
            data_raw[z_min:z_max, y_min:y_max, x_min:x_max],
            raw_spacing,
            new_spacing,
            crop_size,
        )
        save_VOI(
            savepath[1],
            num,
            z_min,
            y_min,
            x_min,
            z_max,
            y_max,
            x_max,
            raw_spacing[0],
            raw_spacing[1],
            raw_spacing[2],
        )
        np.save(os.path.join(savepath[1], num + "_raw_256.npy"), data_raw_crop)
        print(num + " raw", data_raw_crop.shape, data_raw_crop.dtype)
        time_elapsed = time.time() - start
        print(
            "Data sample1 run {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        #### 数据降噪（可选择）,存储时转换成16位 ######################
        if denoise:
            start = time.time()
            data_temp = data_raw_crop.copy()
            mask = data_temp > 0
            sigma = estimate_sigma(data_temp, N=64)
            data_dns_crop = nlmeans(
                data_temp,
                sigma=sigma,
                mask=mask,
                patch_radius=2,
                block_radius=4,
                rician=False,
            )
            data_dns_crop = data_dns_crop.astype(np.int16)
            np.save(
                os.path.join(savepath[1], num + "_denoised_256.npy"), data_dns_crop
            )  #### int16 clean数据
            print(num + " dns", data_dns_crop.shape, data_dns_crop.dtype)
            time_elapsed = time.time() - start
            print(
                "Data sample1 denoising run {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )


def labels_preprocessing_and_save(all_cases_dir, savepath, crop_size, new_spacing):
    ml_lists, tl_lists, fl_lists = get_all_labelpaths(
        all_cases_dir
    )  #### 数据存储路径：./0005/0005-TL(FL,mask)/IM000.dcm
    for index, (ml_path, tl_path, fl_path) in enumerate(
        zip(ml_lists, tl_lists, fl_lists)
    ):
        #### 从label子路径中读取数据，存储为label_512 #######
        num = ml_path.split("/")[-2]
        label, _, _ = load_itk_image(ml_path, 0)
        labelT, _, _ = load_itk_image(tl_path, 0)
        labelF, _, _ = load_itk_image(fl_path, 0)
        label = label.astype(np.uint8)
        labelT = labelT.astype(np.uint8)
        labelF = labelF.astype(np.uint8)
        labelB = label - labelF - labelT
        temp = measure.label(labelB, connectivity=1)
        regionlist = sorted(
            measure.regionprops(temp), key=lambda x: x.area, reverse=True
        )
        for i in range(len(regionlist)):
            if regionlist[i].area < 700:
                labelB[
                    regionlist[i].coords[:, 0],
                    regionlist[i].coords[:, 1],
                    regionlist[i].coords[:, 2],
                ] = 0
        labelM = label - labelB
        np.save(os.path.join(savepath[0], num + "_L512.npy"), label)
        np.save(os.path.join(savepath[0], num + "_T512.npy"), labelT)
        np.save(os.path.join(savepath[0], num + "_F512.npy"), labelF)
        np.save(os.path.join(savepath[0], num + "_M512.npy"), labelM)
        np.save(os.path.join(savepath[0], num + "_B512.npy"), labelB)
        print(index, num, label.shape, label.dtype)

        #### 提取数据的VOI区域，数据截取并存储 #####
        l_p, h_p, raw_spacing = read_VOIrange_spacing(
            os.path.join(savepath[1], num + "_VOI.txt")
        )
        labelM_crop = VOI_resample(
            labelM[l_p[0] : h_p[0], l_p[1] : h_p[1], l_p[2] : h_p[2]],
            raw_spacing,
            new_spacing,
            crop_size,
            order=1,
        )
        labelT_crop = VOI_resample(
            labelT[l_p[0] : h_p[0], l_p[1] : h_p[1], l_p[2] : h_p[2]],
            raw_spacing,
            new_spacing,
            crop_size,
            order=1,
        )
        labelF_crop = VOI_resample(
            labelF[l_p[0] : h_p[0], l_p[1] : h_p[1], l_p[2] : h_p[2]],
            raw_spacing,
            new_spacing,
            crop_size,
            order=1,
        )
        labelB_crop = VOI_resample(
            labelB[l_p[0] : h_p[0], l_p[1] : h_p[1], l_p[2] : h_p[2]],
            raw_spacing,
            new_spacing,
            crop_size,
            order=1,
        )
        labelM_crop[labelM_crop > 0] = 1
        labelM_crop = labelM_crop.astype(np.uint8)
        labelT_crop[labelT_crop > 0] = 1
        labelT_crop = labelT_crop.astype(np.uint8)
        labelF_crop[labelF_crop > 0] = 1
        labelF_crop = labelF_crop.astype(np.uint8)
        labelB_crop[labelB_crop > 0] = 1
        labelB_crop = labelB_crop.astype(np.uint8)
        labelMB_crop = labelB_crop.copy() * 2
        labelMB_crop[labelM_crop > 0] = 1
        labelTF_crop = labelF_crop.copy() * 2
        labelTF_crop[labelT_crop > 0] = 1
        labelTFB_crop = labelB_crop.copy() * 3
        labelTFB_crop[labelF_crop > 0] = 2
        labelTFB_crop[labelT_crop > 0] = 1
        np.save(os.path.join(savepath[1], num + "_M256.npy"), labelM_crop)
        np.save(os.path.join(savepath[1], num + "_T256.npy"), labelT_crop)
        np.save(os.path.join(savepath[1], num + "_F256.npy"), labelF_crop)
        np.save(os.path.join(savepath[1], num + "_MB256.npy"), labelMB_crop)
        np.save(os.path.join(savepath[1], num + "_TF256.npy"), labelTF_crop)
        np.save(os.path.join(savepath[1], num + "_TFB256.npy"), labelTFB_crop)
        print(index, num, labelTFB_crop.shape, labelTFB_crop.dtype)


def pred_and_concate(model, newdata, crop_z=96):
    #### newdata has the resolution of n*256*256 #####
    prediction_bg = np.zeros_like(newdata)
    prediction_ad = np.zeros_like(newdata)
    prediction_br = np.zeros_like(newdata)
    for i in range(int(newdata.shape[0] / crop_z)):
        savedata = newdata[i * crop_z : (i + 1) * crop_z]
        savedata = torch.from_numpy(savedata[np.newaxis, np.newaxis])
        savedata = savedata.float().cuda()
        with torch.no_grad():
            output = model(savedata)
        prediction_bg[i * crop_z : (i + 1) * crop_z] = (
            output.detach().cpu().numpy()[0, 0]
        )
        prediction_ad[i * crop_z : (i + 1) * crop_z] = (
            output.detach().cpu().numpy()[0, 1]
        )
        prediction_br[i * crop_z : (i + 1) * crop_z] = (
            output.detach().cpu().numpy()[0, 2]
        )
    savedata = np.zeros((crop_z, 256, 256))
    savedata[0 : (newdata.shape[0] - (i + 1) * crop_z)] = newdata[(i + 1) * crop_z :]
    savedata = torch.from_numpy(savedata[np.newaxis, np.newaxis])
    savedata = savedata.float().cuda()
    with torch.no_grad():
        output = model(savedata)
    prediction_bg[(i + 1) * crop_z :] = (
        output.detach().cpu().numpy()[0, 0, 0 : (newdata.shape[0] - (i + 1) * crop_z)]
    )
    prediction_ad[(i + 1) * crop_z :] = (
        output.detach().cpu().numpy()[0, 1, 0 : (newdata.shape[0] - (i + 1) * crop_z)]
    )
    prediction_br[(i + 1) * crop_z :] = (
        output.detach().cpu().numpy()[0, 2, 0 : (newdata.shape[0] - (i + 1) * crop_z)]
    )
    prediction = np.concatenate(
        (
            prediction_bg[np.newaxis, np.newaxis],
            prediction_ad[np.newaxis, np.newaxis],
            prediction_br[np.newaxis, np.newaxis],
        ),
        1,
    )
    prediction = np.argmax(prediction, axis=1)
    return prediction


if __name__ == "__main__":
    ####### 获取数据目录下的所有的case的dicom路径#################
    all_cases_dir = "/home/train/ADDATA/20181213_AllCases"  # save raw dicoms
    save_dir = [
        "/home/train/tmp/data_512",
        "/ssd/data_256",
    ]  # save 512*512 and 256*256 datas
    crop_size = 256
    new_spacing = np.array([1, 0.7, 0.7])
    denoise = False
    ######## 处理数据 ###############################################
    data_preprocessing_and_save(
        all_cases_dir,
        savepath=save_dir,
        crop_size=crop_size,
        new_spacing=new_spacing,
        denoise=denoise,
    )
    # labels_preprocessing_and_save(all_cases_dir, savepath=save_dir, crop_size=crop_size, new_spacing=new_spacing)
