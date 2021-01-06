"""
合并mvd6,7,8全部用于训练
"""

import os
import morphtorch as MT
import numpy as np
# import ipyvolume as ipv
from matplotlib import pyplot as plt
import re
import pydicom
import SimpleITK as sitk
import zipfile
import json
from scipy import ndimage
import random
import shutil
import pydicom as dcm
from skimage.measure import label as label_connect
from scipy.ndimage.morphology import binary_dilation
from tqdm import tqdm
from scipy.ndimage import zoom
from skimage import filters
import multiprocessing
from imshow3d import ImShow3D

# 文件后缀，归一化后
img_subfix = "_new_raw.npy"
lab_subfix = "_new_liver_lab.npy"

# 原始压缩文件路径
# SOURCE_DIR = ["/yupeng/biz_rawdata/2c9180947481099601768845926930e3",
#               "/yupeng/biz_rawdata/2c91809474810996017688469ef131b6",
#               "/yupeng/biz_rawdata/2c9180947481099601768847e41c3257",
#              ]
SOURCE_DIR = ["/yupeng/biz_rawdata/2c9180947481099601768981eb5b34d7"]

# 所有数据父级目录
ROOT_DIR = "/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/"

# 解压缩后文件路径
UNZIP_DIR = "/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/MVD9" 
# img和label文件路径
DATA_DIR = "/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/mvd9_t2" # 原始npy
LIVER_DIR = "/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/mvd9_t2_liver" # 归一化后liver
ORGAN_DIR = "/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/mvd9_t2_organ" # 归一化liver + spleen
# txt路径
LIVER_TXT = "/ssd/Jupiter/organ_seg/organ_seg_splits/20201222_mvd9_liver.txt"
ORGAN_TXT = "/ssd/Jupiter/organ_seg/organ_seg_splits/20201222_mvd9_organ.txt"

# x, y spacing
XY_BASE_SPACING = [1, 1]

# 厚层z方向抽取系数
Z_THICKNESS_NORM = 7
INTERPOLATE_THRESH = 6


def do_thickness_norm(raw_img, raw_label, raw_thickness):
    """
    抽取z轴的数据
    :param raw_img: 
    :param raw_label: 
    :param raw_thickness: 

    """
    raw_slice = np.shape(raw_img)[0]
    new_slice_num = round(raw_thickness * raw_slice / Z_THICKNESS_NORM)
    new_img, new_label = [], []
    for new_idx in np.around(np.linspace(0, raw_slice-1, new_slice_num)).astype(int):
        new_img.append(raw_img[new_idx])
        new_label.append(raw_label[new_idx])

    new_img = np.stack(new_img, axis=0)
    new_label = np.stack(new_label, axis=0)
    print('=====Original thickness {}\toriginal shape {}\tnew shape {}'.format(raw_thickness, raw_img.shape, new_img.shape))
    return new_img, new_label


def jupiter_get_img(path: str) -> (np.ndarray, np.ndarray):
    """
    读取一个case的img文件
    :param path: 
    
    """

    # sort and align raw images
    liver_slices = [
        pydicom.read_file(os.path.join(path, file)) for file in os.listdir(path)
    ]
    liver_slices.sort(key=lambda x: x.ImagePositionPatient[2])

    dicom_template = liver_slices[0]
    z_spacing = abs(
        liver_slices[0].ImagePositionPatient[-1]
        - liver_slices[-1].ImagePositionPatient[-1]
    ) / (len(liver_slices) - 1)
    x_spacing = float(dicom_template.PixelSpacing[1])
    y_spacing = float(dicom_template.PixelSpacing[0])
    # try: 
    liver_slices = np.stack([sli.pixel_array for sli in liver_slices], axis=0).astype(np.float)
    # except:
    #     print('Error: {}'.format(path))
    #     liver_slice = np.array([])
    #     # exit()

    return liver_slices, np.array((z_spacing, y_spacing, x_spacing))


def jupiter_get_mask(ann_sub_dir) -> dict():
    """
    读取一个case的mask文件，包含liver和spleen
    :param ann_sub_dir: 某一个case的标签路径

    return: 
        maskType2npy -> dict: 不同脏器的mask ndarray

    """
    if not os.path.exists(ann_sub_dir):
        print('Anote Error!\t{}'.format(ann_sub_dir))
        return {}

    ann_sub_dir_list = sorted(os.listdir(ann_sub_dir))

    # 无标注
    if 'no_seg.zip' in ann_sub_dir_list:
        return {}

    maskType2npy = {}
    for f in ann_sub_dir_list:
        # 需要.zip和spleen & liver的文件
        if not f.endswith('.zip'):
            continue
        if not (f.startswith('spleen') or f.startswith('liver')):
            continue
        
        mask_type_zip = os.path.join(ann_sub_dir, f)
        mask_type = os.path.splitext(f)[0] # liver or spline
        
        # print(mask_type_zip)
        curr_unzip_path = mask_type_zip.replace('.zip', '')  # 现有的解压后的文件

        # 删除原始已经解压后的文件，只保留未解压的.zip
        if os.path.exists(curr_unzip_path):
            shutil.rmtree(curr_unzip_path)
        
        # 解压.zip
        # 结尾出现空格的情况
        if mask_type_zip.split('.')[0][-1] == " ":
            os.system('unzip {} -d {} > /dev/null 2>&1'.format(mask_type_zip.split('.')[0].strip() + '\ ', ann_sub_dir))
            print('Error name\t case:{}'.format(mask_type_zip))
        else:
            os.system('unzip {} -d {} > /dev/null 2>&1'.format(mask_type_zip, ann_sub_dir))

        # os.system('unzip {} -d {}> /dev/null 2>&1'.format(mask_type_zip, ann_sub_dir))
        # 对解压后的文件夹内文件排序
        while len(os.listdir(curr_unzip_path)) == 1:
            tmp_name = os.listdir(curr_unzip_path)[0]
            curr_unzip_path = os.path.join(curr_unzip_path, tmp_name)

        sorted_mask_dir = sorted(os.listdir(curr_unzip_path))
        dir_name = curr_unzip_path

        # 多个合并
        mask_list = []
        for i in range(len(sorted_mask_dir)):
            mask_dcm = pydicom.read_file(os.path.join(dir_name, sorted_mask_dir[i]))
            mask_np = mask_dcm.pixel_array
            mask_list.append(mask_np)

        mask = np.stack(mask_list, axis=0)
        # 检查是否有 XX.zip 和 XX001.zip 同时存在的情况, 以 XX001.zip 为标准
        if re.sub(r'00\d', '', mask_type) in maskType2npy.keys():
            maskType2npy[re.sub(r'00\d', '', mask_type).strip()] = mask
        else:
            maskType2npy[mask_type.strip()] = mask
        
    return maskType2npy


def get_noramlize_organ(spacing_dict):
    """
    对于x,y进行缩放，z进行重采样
    
    """

    caseids = [i.replace('_img.npy', '').strip() for i in os.listdir(DATA_DIR) if i.endswith('_img.npy')]
    print(len(caseids))

    for c in tqdm(caseids, total=len(caseids)):
        img = np.load(os.path.join(DATA_DIR, '{}_img.npy'.format(c)))
        spleen_mask = np.load(os.path.join(DATA_DIR, '{}_label.npy'.format(c))).astype(np.uint8)
        if np.shape(img) != np.shape(spleen_mask):
            print('=====Mask Error!\t{} - {} Spleen mask Not Equal to Liver mask!====='.format(mvd, c))
            continue
        # spleen_mask = np.where(spleen_mask>0, 1, 0).astype(np.uint8)
        
        if np.max(spleen_mask) != 2:
            print('=====Lab Error!\t{} Mask lab is Wrong!====='.format(c))
            continue
        
        curr_spacing = np.array(spacing_dict[c])
        zoom_sacle = curr_spacing[1:] / XY_BASE_SPACING
        zoom_sacle = np.insert(zoom_sacle, 0, 1.0)
        assert zoom_sacle[0] == 1.0
        # print('{} spacing is {}\tscale is {}'.format(c, curr_spacing, zoom_sacle))
        
        reshape_img = zoom(img.astype(np.float), zoom_sacle, order=3)
        reshape_label = zoom(spleen_mask, zoom_sacle, order=0)
        
        z_thickness = curr_spacing[0]
        if z_thickness < INTERPOLATE_THRESH:
            reshape_img, reshape_label = do_thickness_norm(raw_img=reshape_img, raw_label=reshape_label, raw_thickness=z_thickness)
        
        # np.save(os.path.join(spleen_ssd_dir, '{}_img.npy'.format(c)), img.astype(np.int16))
        # np.save(os.path.join(spleen_ssd_dir, '{}_label.npy'.format(c)), spleen_mask.astype(np.uint8))
        
        # print('Original shape {}\tNew shape {}\t{}'.format(img.shape, reshape_img.shape, count))
        np.save(os.path.join(ORGAN_DIR, '{}{}'.format(c, img_subfix)), reshape_img.astype(np.int16))
        np.save(os.path.join(ORGAN_DIR, '{}{}'.format(c, lab_subfix)), reshape_label.astype(np.uint8))
        # count += 1

def get_noramlize_liver(spacing_dict):
    """
    对于x,y进行缩放，z进行重采样
    
    """

    caseids = [i.replace('_img.npy', '').strip() for i in os.listdir(DATA_DIR) if i.endswith('_img.npy')]
    print(len(caseids))

    for c in tqdm(caseids, total=len(caseids)):
        img = np.load(os.path.join(DATA_DIR, '{}_img.npy'.format(c)))
        spleen_mask = np.load(os.path.join(DATA_DIR, '{}_label.npy'.format(c))).astype(np.uint8)
        spleen_mask[spleen_mask == 2] = 0
        if np.shape(img) != np.shape(spleen_mask):
            print('=====Mask Error!\t{} - {} Spleen mask Not Equal to Liver mask!====='.format(mvd, c))
            continue
        # spleen_mask = np.where(spleen_mask>0, 1, 0).astype(np.uint8)
        
        if np.max(spleen_mask) != 1:
            print('=====Lab Error!\t{} Mask lab is Wrong!====='.format(c))
            continue
        
        curr_spacing = np.array(spacing_dict[c])
        zoom_sacle = curr_spacing[1:] / XY_BASE_SPACING
        zoom_sacle = np.insert(zoom_sacle, 0, 1.0)
        assert zoom_sacle[0] == 1.0
        # print('{} spacing is {}\tscale is {}'.format(c, curr_spacing, zoom_sacle))
        
        reshape_img = zoom(img.astype(np.float), zoom_sacle, order=3)
        reshape_label = zoom(spleen_mask, zoom_sacle, order=0)
        
        z_thickness = curr_spacing[0]
        if z_thickness < INTERPOLATE_THRESH:
            reshape_img, reshape_label = do_thickness_norm(raw_img=reshape_img, raw_label=reshape_label, raw_thickness=z_thickness)
        
        # np.save(os.path.join(spleen_ssd_dir, '{}_img.npy'.format(c)), img.astype(np.int16))
        # np.save(os.path.join(spleen_ssd_dir, '{}_label.npy'.format(c)), spleen_mask.astype(np.uint8))
        
        # print('Original shape {}\tNew shape {}\t{}'.format(img.shape, reshape_img.shape, count))
        np.save(os.path.join(LIVER_DIR, '{}{}'.format(c, img_subfix)), reshape_img.astype(np.int16))
        np.save(os.path.join(LIVER_DIR, '{}{}'.format(c, lab_subfix)), reshape_label.astype(np.uint8))
        # count += 1


def write_txt(is_liver=True):
    """
    路径写入txt文件
    :param is_liver -> bool: 
    
    """
    if is_liver:
        cur_dir = LIVER_DIR
        cur_txt_dir = LIVER_TXT
    else:
        cur_dir = ORGAN_DIR
        cur_txt_dir = ORGAN_TXT
        
    files = os.listdir(cur_dir)
    pre_dir = cur_dir.split('/')[-1]

    id_set = set()
    for i in files:
        if 'DI' in i:
            name = os.path.join(pre_dir, 'DI_' + i.split('_')[1])
            if name not in id_set:
                id_set.add(name)
    
    print(len(id_set))
    with open(cur_txt_dir, 'w') as f:
        for i in id_set:
            f.write(i + '\n')


def data_check(is_liver=True):
    """
    检查一个目录下的所有标注文件是否具有相同类型的标注（只有liver一个通道，含有spleen两个通道）
    
    :param is_liver -> bool: 是否检查的是liver部分，或是spleen部分

    """

    if is_liver:
        files = os.listdir(LIVER_DIR)
        # 标注通道数量, liver = 2, spleen = 3
        num_channel = 2
        cur_dir = LIVER_DIR
    else:
        files = os.listdir(ORGAN_DIR)
        num_channel = 3
        cur_dir = ORGAN_DIR

    
    for f in files:
        if 'DI' in f and '_new_liver_lab.npy' in f:
            mask = np.load(os.path.join(cur_dir, f))
            if len(np.unique(mask)) != num_channel:
                print('error mask\tcase: {}'.format('DI_' + f.split('_')[1]))


if __name__ == '__main__':
    
    print(9)

    mvd = 'MVD9'
    error_list_info = []
    spacing_dict = {}

    if not os.path.exists(UNZIP_DIR):
        os.mkdir(UNZIP_DIR)

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    if not os.path.exists(LIVER_DIR):
        os.mkdir(LIVER_DIR)

    if not os.path.exists(ORGAN_DIR):
        os.mkdir(ORGAN_DIR)

    # unzip
    '''
    for s_dir in SOURCE_DIR:
        for f in tqdm(os.listdir(s_dir)):
            if not os.path.exists(os.path.join(UNZIP_DIR, f.split('.')[0])):
                os.system('unzip {} -d {}'.format(os.path.join(s_dir, f), UNZIP_DIR))

    # 获取标注和图像
    
    for case_id in os.listdir(UNZIP_DIR):
        if case_id.startswith('.DS_Store'):
            continue
        
        # liver & spleen mask, 
        mask_type_to_array = jupiter_get_mask(os.path.join(UNZIP_DIR, case_id, 'annotation'))

        # 需要有liver的标注
        if not mask_type_to_array:
            continue
        if 'liver' not in mask_type_to_array: 
            print('Mask Error!\t{} - {} No Liver mask!'.format(mvd, case_id))
            error_list_info += '{} - {}\n'.format(mvd, case_id)
            continue
        
        liver_label = mask_type_to_array['liver']
        if 'spleen' in mask_type_to_array:
            spleen_label = mask_type_to_array['spleen']
            
            if np.shape(spleen_label) != np.shape(liver_label):
                print('Mask Error!\t{} - {} Spleen mask Not Equal to Liver mask!'.format(mvd, c))
                error_list_info += '{} - {}\n'.format(mvd, c)
                continue
            # It is assumed that liver mask is True! Therefore the Spleen mask should be subtracted by Liver mask
            # TODO: Check whether this assumption is Correct!
            spleen_label[liver_label > 0] = 0
            
            # 合并两个label，liver = 1，spleen = 2
            liver_label = liver_label + 2 * spleen_label

        # img & spacing
        img_array, spacing = jupiter_get_img(os.path.join(UNZIP_DIR, case_id, 'slices'))

        spacing_dict[case_id] = spacing.tolist()
        np.save(os.path.join(DATA_DIR, '{}{}'.format(case_id, '_label.npy')), liver_label.astype(np.uint8))
        np.save(os.path.join(DATA_DIR, '{}{}'.format(case_id, '_img.npy')), img_array.astype(np.int16))
    
    with open(os.path.join(ROOT_DIR, 'spacing_mvd9.json'), 'w') as f:
        json.dump(spacing_dict, f)
    
    with open(os.path.join(ROOT_DIR, 'spacing_mvd9.json'), 'r') as f:
        spacing_dict = json.load(f)
    '''
    # 对npy的图像和mask进行大小归一化
    # get_noramlize_liver(spacing_dict)
    # get_noramlize_organ(spacing_dict)
    
    

    # write_txt(is_liver=False)
    data_check(is_liver=True)
    
    '''
    '''