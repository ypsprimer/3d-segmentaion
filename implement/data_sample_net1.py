import numpy as np
import os, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
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
    dataImporter.SetWholeExtent(0, h - 1, 0, d - 1, 0, w - 1)
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


def VOI_recover(
    output_256, saved_voi_range, mode="nearest", order=2
):  ### recover的输入必须是int
    output_256 = output_256.astype(np.uint8)
    resize_factor = saved_voi_range / output_256.shape
    VOI = zoom(output_256, resize_factor, mode=mode, order=order)
    return VOI


def get_all_dicompaths(data_root):
    ################## input: the path of dicom #####################
    #### output: lists of dicompaths and patient names (or No.) #####
    dicompath = data_root
    dicomdirs = [
        os.path.join(dicompath, name)
        for name in os.listdir(dicompath)
        if "dicom" in name
    ]
    dicomdirs.sort()
    titles = [name.split("/")[-1].split("-")[0] for name in dicomdirs]
    return dicomdirs, titles


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
