import torch
import torch.optim as optim
import numpy as np
import os, sys
import ipyvolume
import vtk
import matplotlib.pyplot as plt
import xlwt
import cv2
from math import ceil, floor, sqrt
from skimage import data, filters, feature, segmentation, measure, morphology
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_closing,
    binary_opening,
)
from scipy import ndimage as ndi
from scipy.spatial.distance import directed_hausdorff
from scipy.sparse import csgraph, coo_matrix  # 用来存储稀疏矩阵的
from scipy.interpolate import RegularGridInterpolator
from skimage.morphology import watershed


def resize(
    imgs, size=None, resize_factor=None, backsize=False, mode="nearest", order=1
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


def denoise_lumen(data):
    data = data.astype(np.float)
    tempdata = data.copy()
    regionlist = sorted(
        measure.regionprops(measure.label(tempdata, connectivity=1)),
        key=lambda x: x.area,
        reverse=True,
    )
    for i in range(1, len(regionlist)):
        tempdata[
            regionlist[i].coords[:, 0],
            regionlist[i].coords[:, 1],
            regionlist[i].coords[:, 2],
        ] = 0
    data = (data * tempdata).astype(np.uint8)
    return data


##################################################################################################################
################################################# 结果后处理过程相关函数 #########################################
def containing_score(sub_region, ref_region):
    score = np.sum(sub_region * ref_region) / np.sum(sub_region)
    return score


def get_flap_area(recent_AD_slice):
    AD_slice = recent_AD_slice.copy()  ######### 用膜片来判断是否是到达了底部没有假腔的地方
    close_AD_slice = binary_closing(AD_slice, iterations=5)
    flap = close_AD_slice - AD_slice
    return np.sum(flap)


def get_domains_num(recent_AD_slice):
    measure_slice = measure.label(recent_AD_slice, connectivity=1)
    num = len(measure.regionprops(measure_slice))
    return num


def my_watershed(region1, region2, target):
    lumen = binary_erosion(target, iterations=2)
    overlap = ~((target > 0) & lumen)
    distance = ndi.distance_transform_edt(overlap).astype("float32")  ### 启动距离变换，生成待分离的图
    distance[target != 1] = 0
    distance = np.max(distance) - distance
    distance[target != 1] = 0
    my_marker = region1 + region2
    markers = ndi.label(my_marker)[0]
    labels = watershed(
        -distance, markers, mask=target, watershed_line=False
    )  ### 分离，不管是不是这个层里面有黏连
    return labels


def region_simplify(seg_regions, thresholds=(20, 50)):  ### 这个函数值得好好优化
    threshold = thresholds[0]
    reference_thr = thresholds[1]
    regions = seg_regions.copy()
    region_nums = regions.max()
    if region_nums != 1 and region_nums != 0:
        values = (
            np.argsort(
                np.array([np.sum(regions == num) for num in range(1, region_nums + 1)])
            )
            + 1
        )  ### 根据面积从小到大原则排列标记
        for i in range(values.shape[0]):
            temp_region = regions == values[i]
            if np.sum(temp_region) == 0:
                continue  ### 如果区域面积已经为零，说明已被简并，不再考虑
            elif np.sum(temp_region) >= threshold:
                break  ### 如果区域面积大于小区域上限，我们不再执行简并
            else:  ### 如果区域面积在界限以内，则执行简并
                region_matrices = []  ### 创建一个不同区域的叠层矩阵，每一层都是一个独立的域，按value顺序
                for num in range(1, region_nums + 1):
                    region_matrices.append(regions == num)
                region_matrices = np.array(region_matrices)  ### 到此把所有的区域拆开分别放在了各个层中
                temp_matrices = region_matrices | temp_region  ### 和目前要比较的最小区域做或运算
                for j in range(values.shape[0] - 1, i, -1):  ### 我们要和比当前区域大的区域比较
                    measure_labels = measure.label(
                        temp_matrices[values[j] - 1], connectivity=1
                    )  ### 测量有几个连通域
                    measure_regions = measure.regionprops(measure_labels)
                    if len(measure_regions) == 1:
                        regions[regions == values[i]] = values[j]
                        values = (
                            np.argsort(
                                np.array(
                                    [
                                        np.sum(regions == num)
                                        for num in range(1, region_nums + 1)
                                    ]
                                )
                            )
                            + 1
                        )
                        break
        label = 1
        for num in range(1, regions.max() + 1):  ### 遍历一下图中区域，更改编号
            if np.sum(regions == num) > 0:
                regions[regions == num] = label
                label += 1
    return regions


def TFL_separation(
    Net1_AD,
    Net2_TL,
    Net2_FL,
    area_above_thr=(20, 50),
    area_below_thr=(30, 50),
    sim_thr=0.27,
):
    ############ 这个算法里面有几个参数要注意调节：###################################################################
    ############ 1.当上一层正常划分为两个腔时，当前层只有一个域时，连续层面积变化梯度均值计算中层数默认取5 ###########
    ############ 2.当上一层正常划分为两个腔时，当前层两个域与上一层两个腔分别的覆盖率阈值，默认为0.27 ################
    ############ 3.区域优化时的小区域上限和大区域下限，分别为上行(20,50)，下行（30,50）###############################
    ############ 3.为了避免把升主的区域弄进来，有一个距离阈值，默认设置为50 ##########################################

    result_AD = Net1_AD.copy()
    Network_TL = Net2_TL.copy()
    Network_FL = Net2_FL.copy()
    ##### 找到修正的上行终止层suplim ########
    main_label_close = binary_closing(result_AD, iterations=4)
    z, _, _ = np.where(main_label_close)
    for slice in range(z.max(), -1, -1):
        temp_main = measure.label(main_label_close[slice], connectivity=1)
        regionlist = sorted(
            measure.regionprops(temp_main), key=lambda x: x.area, reverse=True
        )
        if len(regionlist) == 1 and regionlist[0].area > 3000:
            first_slice = slice
            break
    last = 10000
    suplim = first_slice
    for slice in range(first_slice, -1, -1):
        temp_main = measure.label(main_label_close[slice], connectivity=1)
        regionlist = sorted(
            measure.regionprops(temp_main), key=lambda x: x.area, reverse=True
        )
        if len(regionlist) >= 2:
            if (len(regionlist) == 2) & (
                len(regionlist) - last == 1
            ):  ### 这里已经限定当前层是2个域，上一层是1个域
                if (regionlist[0].area > 1000) & (regionlist[1].area > 1000):
                    suplim = slice - 10
                    break
                elif (regionlist[0].area < 100) | (regionlist[1].area < 100):
                    last = 1
            elif (len(regionlist) > 2) & (
                len(regionlist) - last == 1
            ):  ### 如果当前层不是两个域，是3个，上一层是2个域，可能是膜片没有闭合，保留了真假两腔
                if (
                    (regionlist[0].area > 1000)
                    & (regionlist[1].area > 1000)
                    & (regionlist[2].area > 1000)
                ):
                    suplim = slice - 10
                    break
            elif (len(regionlist) > 2) & (
                len(regionlist) - last > 1
            ):  ### 可能的特殊情况，多出了一个小区域，当前层3个，上一层1个
                if (
                    (regionlist[0].area > 1000)
                    & (regionlist[1].area > 1000)
                    & (regionlist[2].area < 200)
                ):
                    suplim = slice - 10
                    break
        else:
            last = len(regionlist)

    ##### 从网络2的AD结果，网络2的TFL中去掉升主动脉 ########
    top_slice = main_label_close[first_slice]  ### 借鉴上一步找上限的时候的起始层
    top_y_coords, top_x_coords = np.where(top_slice)  ### 寻找这个层的前景中位数
    top_y, top_x = (
        int((top_y_coords.max() + top_y_coords.min()) / 2),
        int((top_x_coords.max() + top_x_coords.min()) / 2),
    )
    main_label_close[:, top_y:, :] = 0  ### 降主的那一侧置零
    half_main_maxz = []
    half_main = measure.label(main_label_close, connectivity=1)
    half_main_regions = measure.regionprops(half_main)
    for half_main_region in half_main_regions:
        half_main_maxz.append(half_main_region.coords[:, 0].max())
    half_main_maxz = np.array(half_main_maxz)  ### 我们要找到升主的一侧具有最高Z轴的区域
    maxz_index = np.argmax(half_main_maxz)
    asc_aorta_z = half_main_regions[maxz_index].coords[:, 0]  ### 获取这个区域的坐标
    asc_aorta_y = half_main_regions[maxz_index].coords[:, 1]
    asc_aorta_x = half_main_regions[maxz_index].coords[:, 2]
    result_AD[asc_aorta_z, asc_aorta_y, asc_aorta_x] = 0  ### 然后把三个东西的这个区域全部置零，干掉升主
    Network_TL[asc_aorta_z, asc_aorta_y, asc_aorta_x] = 0
    Network_FL[asc_aorta_z, asc_aorta_y, asc_aorta_x] = 0

    ##### 找到修正的起点层start ########
    start = None
    for i in range(
        int((z.min() + z.max()) / 2), suplim + 1
    ):  # Nei1 AD结果的中间位置到上行终止点，目的是找到分离起始层
        if np.max(result_AD[i]) > 0:  # 判断是不是有前景色（主动脉）
            image = result_AD[i].copy()
            label_img = measure.label(image, connectivity=1)  # 4邻接判断连通性
            for region in measure.regionprops(label_img):  # 去掉小区域以后如果剩余是两个域
                if region.area < 10:
                    image[region.coords[:, 0], region.coords[:, 1]] = 0
            label_img = measure.label(image, connectivity=1)
            if i > (z.max() - z.min()) * 0.6 + z.min():
                if len(measure.regionprops(label_img)) == 2:  # 这里已经包括了正常情况，也包括一升一降的情况
                    image_close = binary_closing(image, iterations=3)
                    label_close = measure.label(image_close, connectivity=1)
                    if (
                        len(measure.regionprops(label_close)) == 1
                    ):  # 闭运算以后发现只有1个域，说明是正常的可选择的层
                        target = result_AD[i].copy()  ### 复制当前层
                        region1 = binary_erosion(
                            Network_TL[i], iterations=1
                        )  ### 得到的真腔，缩小一圈，作为marker
                        region2 = binary_erosion(
                            Network_FL[i], iterations=1
                        )  ### 得到的假腔，缩小一圈，marker
                        seg_regions = my_watershed(
                            region1, region2, target
                        )  ### 不能直接使用，可以先通过一个初步后处理干掉非常小的域
                        if seg_regions.max() == 2:
                            start = i
                            break

    if not start:  ##### 如果上行找不到，我们就下行寻找 #####
        for i in range(
            int((z.min() + z.max()) / 2), -1, -1
        ):  # 一半 - 10层的位置到底部，目的是找到start层
            if np.max(result_AD[i]) > 0:  # 判断是不是有前景色（主动脉）
                image = result_AD[i].copy()
                label_img = measure.label(image, connectivity=1)  # 4邻接判断连通性
                for region in measure.regionprops(label_img):
                    if region.area < 10:
                        image[region.coords[:, 0], region.coords[:, 1]] = 0  # 去掉噪声小区域
                label_img = measure.label(image, connectivity=1)  # 去掉小区域以后如果剩余是两个域
                if len(measure.regionprops(label_img)) == 2:
                    image_close = binary_closing(
                        image, iterations=3
                    )  # 执行闭运算看看是不是会变成1个域
                    label_close = measure.label(image_close, connectivity=1)
                    if (
                        len(measure.regionprops(label_close)) == 1
                    ):  # 闭运算以后发现只有1个域，说明是正常的可选择的层
                        target = result_AD[i].copy()  ### 复制当前层
                        region1 = binary_erosion(
                            Network_TL[i], iterations=1
                        )  ### 得到的真腔，缩小一圈，作为marker
                        region2 = binary_erosion(
                            Network_FL[i], iterations=1
                        )  ### 得到的假腔，缩小一圈，marker
                        seg_regions = my_watershed(
                            region1, region2, target
                        )  ### 不能直接使用，可以先通过一个初步后处理干掉非常小的域
                        if seg_regions.max() == 2:
                            start = i
                            break

    lumen1 = np.zeros_like(result_AD)
    lumen2 = np.zeros_like(result_AD)
    ####### 向上追溯，从起始位置start层开始到上界 ################
    for i in range(start, suplim + 1):
        if np.max(result_AD[i]) > 0:  ####### 确认当前层里面有前景主动脉
            image = result_AD[i].copy()  ### 复制当前层
            region1 = binary_erosion(
                Network_TL[i], iterations=1
            )  ### 得到的真腔，缩小一圈，作为marker
            region2 = binary_erosion(Network_FL[i], iterations=1)  ### 得到的假腔，缩小一圈，marker
            target = image.copy()
            seg_regions = my_watershed(
                region1, region2, target
            )  ### 不能直接使用，可以先通过一个初步后处理干掉非常小的域
            seg_regions = region_simplify(
                seg_regions, thresholds=area_above_thr
            )  ### 区域简化函数，两个参数：原来的region、region面积最小值
            if (
                np.max(lumen1) == 0
            ):  ##### 入果lumen1还是空的，我们这个肯定是起始层，一个域就放进lumen1，两个域就分别放进lumen1和2
                if seg_regions.max() == 1:
                    lumen1[i] = seg_regions
                if seg_regions.max() == 2:
                    lumen1[i] = seg_regions == 1
                    lumen2[i] = seg_regions == 2
            else:  ##### 如果lumen1不是空的，说明相邻的层已经被搞定，把它作为参考来处理当前的层
                if (
                    lumen1[i - 1].max() > 0 and lumen2[i - 1].max() > 0
                ):  ### 需要确定之前的那一层是不是两个腔都有东西: 是的,上一层是准的
                    if seg_regions.max() == 1:  ### 如果当前层发现只有一个域，我们可以猜测有可能是错分造成或者挤压狭窄
                        score1 = containing_score(
                            lumen1[i - 1], seg_regions
                        )  ### 看看和上边那个域比较像，决定了去计算哪个的变化均值
                        score2 = containing_score(lumen2[i - 1], seg_regions)
                        diff_ave = 0  ### 准备比较前5层的变化程度，计算变化均值
                        if score1 > score2:
                            for j in range(2, 7):
                                diff_ave += (
                                    np.abs(
                                        int(np.sum(lumen1[i - j]))
                                        - int(np.sum(lumen1[i - j + 1]))
                                    )
                                    / 5
                                )
                        else:
                            for j in range(2, 7):
                                diff_ave += (
                                    np.abs(
                                        int(np.sum(lumen2[i - j]))
                                        - int(np.sum(lumen2[i - j + 1]))
                                    )
                                    / 5
                                )
                        diff = np.abs(
                            int(np.sum(seg_regions)) - int(np.sum(lumen1[i - 1]))
                        )  ### 再计算一下当前的层的差异
                        if diff > 20 * diff_ave:  ### 如果变化超过往次变化均值的20倍，说明是错分
                            target = image.copy()
                            changed_region1 = target * binary_erosion(
                                lumen1[i - 1], iterations=1
                            )  ### 改变marker的使用，来源于相邻层
                            changed_region2 = target * binary_erosion(
                                lumen2[i - 1], iterations=1
                            )
                            seg_regions = my_watershed(
                                changed_region1, changed_region2, target
                            )
                    elif seg_regions.max() == 2:  ### 经验证发现有的时候有两个腔也不是很好，不能用net2做marker
                        sim_seg1to1 = containing_score(
                            lumen1[i - 1], seg_regions == 1
                        )  ### 打算通过与上一层的相似度来解决问题
                        sim_seg1to2 = containing_score(lumen2[i - 1], seg_regions == 1)
                        sim_seg2to1 = containing_score(lumen1[i - 1], seg_regions == 2)
                        sim_seg2to2 = containing_score(lumen2[i - 1], seg_regions == 2)
                        if np.sum(seg_regions == 1) > np.sum(
                            seg_regions == 2
                        ):  ### 先比较一下两个域中哪个大哪个小
                            if (
                                sim_seg1to1 >= sim_thr
                                and sim_seg1to2 >= sim_thr
                                and sim_seg2to1 < (1 - sim_thr)
                                and sim_seg2to2 < (1 - sim_thr)
                            ):
                                target = image.copy()  ### 若相似度大的都大于0.5，小的都小于0.5，就是错分
                                changed_region1 = target * binary_erosion(
                                    lumen1[i - 1], iterations=1
                                )  ### 改变marker的使用，来源于相邻层
                                changed_region2 = target * binary_erosion(
                                    lumen2[i - 1], iterations=1
                                )
                                seg_regions = my_watershed(
                                    changed_region1, changed_region2, target
                                )
                        elif np.sum(seg_regions == 1) < np.sum(seg_regions == 2):
                            if (
                                sim_seg2to1 >= sim_thr
                                and sim_seg2to2 >= sim_thr
                                and sim_seg1to1 < (1 - sim_thr)
                                and sim_seg1to2 < (1 - sim_thr)
                            ):
                                target = image.copy()  ### 若相似度大的都大于0.5，小的都小于0.5，就是错分
                                changed_region1 = target * binary_erosion(
                                    lumen1[i - 1], iterations=1
                                )  ### 改变marker的使用，来源于相邻层
                                changed_region2 = target * binary_erosion(
                                    lumen2[i - 1], iterations=1
                                )
                                seg_regions = my_watershed(
                                    changed_region1, changed_region2, target
                                )
                    else:  ### 这回是处理有多个域的情况了
                        temp_for_lumen1 = np.zeros(
                            image.shape
                        )  ### 创建两个辅助判别的腔，用来检查基于Net2预测划分的效果
                        temp_for_lumen2 = np.zeros(image.shape)
                        for num in range(1, seg_regions.max() + 1):
                            coord1_y, coord1_x = np.where(seg_regions == num)
                            score1 = containing_score(seg_regions == num, lumen1[i - 1])
                            score2 = containing_score(seg_regions == num, lumen2[i - 1])
                            if score1 > score2:  ### 进行这部分之后分完，后面开始检测
                                temp_for_lumen1[coord1_y, coord1_x] = 1
                            elif score1 < score2:
                                temp_for_lumen2[coord1_y, coord1_x] = 1
                        penetration_to1 = containing_score(
                            lumen1[i - 1], temp_for_lumen2
                        )  ### 这里要比较的是交替渗透率
                        penetration_to2 = containing_score(
                            lumen2[i - 1], temp_for_lumen1
                        )
                        if (
                            penetration_to1 > 0.1 or penetration_to2 > 0.1
                        ):  ### 如果大于渗透率有任何一个方向大于0.1（默认值），就是不好，换marker
                            target = image.copy()  ### 若相似度大的都大于0.5，小的都小于0.5，就是错分
                            changed_region1 = target * binary_erosion(
                                lumen1[i - 1], iterations=1
                            )  ### 改变marker的使用，来源于相邻层
                            changed_region2 = target * binary_erosion(
                                lumen2[i - 1], iterations=1
                            )
                            seg_regions = my_watershed(
                                changed_region1, changed_region2, target
                            )
                    for num in range(
                        1, seg_regions.max() + 1
                    ):  ### 目前我们直接采用遍历里面的所有的小区域，可能要优化
                        coord1_y, coord1_x = np.where(seg_regions == num)
                        score1 = containing_score(seg_regions == num, lumen1[i - 1])
                        score2 = containing_score(seg_regions == num, lumen2[i - 1])
                        if score1 > score2:
                            lumen1[
                                i, coord1_y, coord1_x
                            ] = 1  ### 要注意这个地方的操作是会忽略掉与上层覆盖率为0的区域的
                        elif score1 < score2:
                            lumen2[i, coord1_y, coord1_x] = 1
                else:  ### 也可能上一层中只有一个腔有东西，发生了腔体压缩缺失/失去真腔形态的错分，这里要优化
                    if seg_regions.max() == 1:  ### 那么当：当前层也只有1个腔
                        coord1_y, coord1_x = np.where(seg_regions)
                        score1 = containing_score(
                            seg_regions, lumen1[i - 1]
                        )  ### 直接比直接分
                        score2 = containing_score(seg_regions, lumen2[i - 1])
                        if score1 > score2:
                            lumen1[i, coord1_y, coord1_x] = 1
                        else:
                            lumen2[i, coord1_y, coord1_x] = 1
                    elif seg_regions.max() == 2:  ### 当前层有2个腔的时候
                        if lumen1[i - 1].max():  ### 看看是哪一个腔有东西: 腔1有东西
                            score1 = containing_score(
                                lumen1[i - 1], seg_regions == 1
                            )  ### 决定把和上一层长得比较像的给它，和之前"包含"策略的理念是反过来的
                            score2 = containing_score(lumen1[i - 1], seg_regions == 2)
                            if score1 > score2:  ### 如果第一个区域长得和上一层像,就把这个给出去
                                lumen1[i] = seg_regions == 1
                                remain_y, remain_x = np.where(seg_regions == 2)
                                remain_y_mean, remain_x_mean = (
                                    np.mean(remain_y),
                                    np.mean(remain_x),
                                )
                                remain_center = np.array(
                                    [[remain_y_mean, remain_x_mean]]
                                )
                                lumen1_y, lumen1_x = np.where(lumen1[i])
                                lumen1_coords = np.array([lumen1_y, lumen1_x]).T
                                dis = directed_hausdorff(remain_center, lumen1_coords)[
                                    0
                                ]  ### 豪斯德夫距离计算当前层的两个域距离是不是远，太远的话另外的那个域就先不要了
                                if dis < 50:  ### 如果距离不是很远的话，也不能轻易把剩余的分到另外一个腔
                                    temp_slice = image.copy()
                                    area_num = get_domains_num(
                                        temp_slice
                                    )  ### 这时需要参考AD的结果里面有几个域
                                    if area_num > 1:  ### 如果AD里面不止有一个域：
                                        lumen2[i] = seg_regions == 2  ### 直接放进第二个区域里面
                                    else:  ### 如果AD里面只有一个域：
                                        flap_area = get_flap_area(
                                            temp_slice
                                        )  ### 测试一下flap的面积
                                        if flap_area < 20:  ### 如果flap很小，这里经验化设置为15
                                            lumen1[i] = (seg_regions == 1) + (
                                                seg_regions == 2
                                            )  ### 就属于错分，不能落下，同样纳入到第一个区域里面
                                        else:
                                            lumen2[i] = (
                                                seg_regions == 2
                                            )  ### 否则再放进第二个区域里面
                            else:
                                lumen1[i] = seg_regions == 2
                                remain_y, remain_x = np.where(seg_regions == 1)
                                remain_y_mean, remain_x_mean = (
                                    np.mean(remain_y),
                                    np.mean(remain_x),
                                )
                                remain_center = np.array(
                                    [[remain_y_mean, remain_x_mean]]
                                )
                                lumen1_y, lumen1_x = np.where(lumen1[i])
                                lumen1_coords = np.array([lumen1_y, lumen1_x]).T
                                dis = directed_hausdorff(remain_center, lumen1_coords)[
                                    0
                                ]
                                if dis < 50:  ### 如果距离不是很远的话，也不能轻易把剩余的分到另外一个腔
                                    temp_slice = image.copy()
                                    area_num = get_domains_num(
                                        temp_slice
                                    )  ### 这时需要参考AD的结果里面有几个域
                                    if area_num > 1:  ### 如果AD里面不止有一个域：
                                        lumen2[i] = seg_regions == 1  ### 直接放进第二个区域里面
                                    else:  ### 如果AD里面只有一个域：
                                        flap_area = get_flap_area(
                                            temp_slice
                                        )  ### 测试一下flap的面积
                                        if flap_area < 20:  ### 如果flap很小，这里经验化设置为15
                                            lumen1[i] = (seg_regions == 1) + (
                                                seg_regions == 2
                                            )  ### 就属于错分，不能落下，同样纳入到第一个区域里面
                                        else:
                                            lumen2[i] = (
                                                seg_regions == 1
                                            )  ### 否则再放进第二个区域里面
                        else:  ############### 腔2有东西的话，道理是一样的：
                            score1 = containing_score(lumen2[i - 1], seg_regions == 1)
                            score2 = containing_score(lumen2[i - 1], seg_regions == 2)
                            if score1 > score2:  ### 如果第一个区域长得和上一层像,就把这个给出去
                                lumen2[i] = seg_regions == 1
                                remain_y, remain_x = np.where(seg_regions == 2)
                                remain_y_mean, remain_x_mean = (
                                    np.mean(remain_y),
                                    np.mean(remain_x),
                                )
                                remain_center = np.array(
                                    [[remain_y_mean, remain_x_mean]]
                                )
                                lumen2_y, lumen2_x = np.where(lumen2[i])
                                lumen2_coords = np.array([lumen2_y, lumen2_x]).T
                                dis = directed_hausdorff(remain_center, lumen2_coords)[
                                    0
                                ]  ### 豪斯德夫距离计算当前层的两个域距离是不是远，太远的话另外的那个域就先不要了
                                if dis < 50:  ### 如果距离不是很远的话，也不能轻易把剩余的分到另外一个腔
                                    temp_slice = image.copy()
                                    area_num = get_domains_num(
                                        temp_slice
                                    )  ### 这时需要参考AD的结果里面有几个域
                                    if area_num > 1:  ### 如果AD里面不止有一个域：
                                        lumen1[i] = seg_regions == 2  ### 直接放进第二个区域里面
                                    else:  ### 如果AD里面只有一个域：
                                        flap_area = get_flap_area(
                                            temp_slice
                                        )  ### 测试一下flap的面积
                                        if flap_area < 20:  ### 如果flap很小，这里经验化设置为15
                                            lumen2[i] = (seg_regions == 1) + (
                                                seg_regions == 2
                                            )  ### 就属于错分，不能落下，同样纳入到第一个区域里面
                                        else:
                                            lumen1[i] = (
                                                seg_regions == 2
                                            )  ### 否则再放进第二个区域里面
                            else:
                                lumen2[i] = seg_regions == 2
                                remain_y, remain_x = np.where(seg_regions == 1)
                                remain_y_mean, remain_x_mean = (
                                    np.mean(remain_y),
                                    np.mean(remain_x),
                                )
                                remain_center = np.array(
                                    [[remain_y_mean, remain_x_mean]]
                                )
                                lumen2_y, lumen2_x = np.where(lumen2[i])
                                lumen2_coords = np.array([lumen2_y, lumen2_x]).T
                                dis = directed_hausdorff(remain_center, lumen2_coords)[
                                    0
                                ]
                                if dis < 50:  ### 如果距离不是很远的话，也不能轻易把剩余的分到另外一个腔
                                    temp_slice = image.copy()
                                    area_num = get_domains_num(
                                        temp_slice
                                    )  ### 这时需要参考AD的结果里面有几个域
                                    if area_num > 1:  ### 如果AD里面不止有一个域：
                                        lumen1[i] = seg_regions == 1  ### 直接放进第二个区域里面
                                    else:  ### 如果AD里面只有一个域：
                                        flap_area = get_flap_area(
                                            temp_slice
                                        )  ### 测试一下flap的面积
                                        if flap_area < 20:  ### 如果flap很小，这里经验化设置为15
                                            lumen2[i] = (seg_regions == 1) + (
                                                seg_regions == 2
                                            )  ### 就属于错分，不能落下，同样纳入到第一个区域里面
                                        else:
                                            lumen1[i] = (
                                                seg_regions == 1
                                            )  ### 否则再放进第二个区域里面
                    else:  ### 最糟糕的情况是，上一层错分/挤压只有一个域，当前层却有好多域
                        for num in range(
                            1, seg_regions.max() + 1
                        ):  ### 这里还没有想出好办法解决，先来个最糙的试试
                            coord1_y, coord1_x = np.where(seg_regions == num)
                            score1 = containing_score(seg_regions == num, lumen1[i - 1])
                            score2 = containing_score(seg_regions == num, lumen2[i - 1])
                            if score1 > score2:
                                lumen1[i, coord1_y, coord1_x] = 1
                            elif score1 < score2:
                                lumen2[i, coord1_y, coord1_x] = 1

    ############ 向下追溯 #################
    for i in range(start - 1, z.min() - 1, -1):
        if np.max(result_AD[i]) > 0:  ####### 确认当前层里面有前景主动脉
            image = result_AD[i].copy()  ### 复制当前层
            region1 = binary_erosion(
                Network_TL[i], iterations=1
            )  ### 得到的真腔，缩小一圈，作为marker
            region2 = binary_erosion(Network_FL[i], iterations=1)  ### 得到的假腔，缩小一圈，marker
            target = image.copy()
            seg_regions = my_watershed(region1, region2, target)
            seg_regions = region_simplify(
                seg_regions, thresholds=area_below_thr
            )  ### 区域简化函数，两个参数：原来的region、region面积最小值
            if (
                np.max(lumen1) == 0
            ):  ##### 入果lumen1还是空的，我们这个肯定是起始层，一个域就放进lumen1，两个域就分别放进lumen1和2
                if seg_regions.max() == 1:
                    lumen1[i] = seg_regions
                if seg_regions.max() == 2:
                    coord1_y, coord1_x = np.where(seg_regions == 1)
                    coord2_y, coord2_x = np.where(seg_regions == 2)
                    lumen1[i, coord1_y, coord1_x] = 1
                    lumen2[i, coord2_y, coord2_x] = 1
            else:  ####################################### 如果lumen1不是空的，说明相邻的层已经被搞定，把它作为参考来处理当前的层
                if (
                    lumen1[i + 1].max() > 0 and lumen2[i + 1].max() > 0
                ):  ### 需要确定之前的那一层是不是两个腔都有东西: 是的
                    if (
                        seg_regions.max() == 1
                    ):  ######### 如果当前层发现只有一个域，我们可以猜测有可能是错分造成或者挤压狭窄
                        score1 = containing_score(
                            lumen1[i + 1], seg_regions
                        )  ### 看看和上边那个域比较像，决定了去计算哪个的变化均值
                        score2 = containing_score(lumen2[i + 1], seg_regions)
                        diff_ave = 0  ### 准备比较前5层的变化程度，计算变化均值
                        if score1 > score2:
                            for j in range(2, 7):
                                diff_ave += (
                                    np.abs(
                                        int(np.sum(lumen1[i + j]))
                                        - int(np.sum(lumen1[i + j - 1]))
                                    )
                                    / 5
                                )
                        else:
                            for j in range(2, 7):
                                diff_ave += (
                                    np.abs(
                                        int(np.sum(lumen2[i + j]))
                                        - int(np.sum(lumen2[i + j - 1]))
                                    )
                                    / 5
                                )
                        diff = np.abs(
                            int(np.sum(seg_regions)) - int(np.sum(lumen1[i + 1]))
                        )  ### 再计算一下当前的层的差异
                        if diff > 10 * diff_ave:  ### 如果变化超过往次变化均值的20倍，说明是错分
                            target = image.copy()
                            changed_region1 = target * binary_erosion(
                                lumen1[i + 1], iterations=1
                            )  ### 改变marker的使用，来源于相邻层
                            changed_region2 = target * binary_erosion(
                                lumen2[i + 1], iterations=1
                            )
                            seg_regions = my_watershed(
                                changed_region1, changed_region2, target
                            )
                    elif (
                        seg_regions.max() == 2
                    ):  ######### 经验证发现有的时候有两个腔也不是很好，不能用net2做marker
                        AD_slice = image.copy()  ######### 用膜片来判断是否是到达了底部没有假腔的地方
                        close_AD_slice = binary_closing(AD_slice, iterations=5)
                        flap = close_AD_slice - AD_slice
                        if np.sum(flap) >= 15:
                            sim_seg1to1 = containing_score(
                                lumen1[i + 1], seg_regions == 1
                            )  ### 打算通过与上一层的相似度来解决问题
                            sim_seg1to2 = containing_score(
                                lumen2[i + 1], seg_regions == 1
                            )
                            sim_seg2to1 = containing_score(
                                lumen1[i + 1], seg_regions == 2
                            )
                            sim_seg2to2 = containing_score(
                                lumen2[i + 1], seg_regions == 2
                            )
                            if np.sum(seg_regions == 1) > np.sum(
                                seg_regions == 2
                            ):  ### 先比较一下两个域中哪个大哪个小
                                if (
                                    sim_seg1to1 >= sim_thr
                                    and sim_seg1to2 >= sim_thr
                                    and sim_seg2to1 < (1 - sim_thr)
                                    and sim_seg2to2 < (1 - sim_thr)
                                ):
                                    target = image.copy()  ### 若相似度大的大于0.5，小的小于0.5，就是错分
                                    changed_region1 = target * binary_erosion(
                                        lumen1[i + 1], iterations=1
                                    )  ### 改变marker的使用，来源于相邻层
                                    changed_region2 = target * binary_erosion(
                                        lumen2[i + 1], iterations=1
                                    )
                                    seg_regions = my_watershed(
                                        changed_region1, changed_region2, target
                                    )
                            elif np.sum(seg_regions == 1) < np.sum(seg_regions == 2):
                                if (
                                    sim_seg2to1 >= sim_thr
                                    and sim_seg2to2 >= sim_thr
                                    and sim_seg1to1 < (1 - sim_thr)
                                    and sim_seg1to2 < (1 - sim_thr)
                                ):
                                    target = image.copy()  ### 若相似度大的大于0.5，小的小于0.5，就是错分
                                    changed_region1 = target * binary_erosion(
                                        lumen1[i + 1], iterations=1
                                    )  ### 改变marker的使用，来源于相邻层
                                    changed_region2 = target * binary_erosion(
                                        lumen2[i + 1], iterations=1
                                    )
                                    seg_regions = my_watershed(
                                        changed_region1, changed_region2, target
                                    )
                    for num in range(
                        1, seg_regions.max() + 1
                    ):  ### 目前我们直接采用遍历里面的所有的小区域，可能要优化
                        coord1_y, coord1_x = np.where(seg_regions == num)
                        score1 = containing_score(seg_regions == num, lumen1[i + 1])
                        score2 = containing_score(seg_regions == num, lumen2[i + 1])
                        if score1 > score2:
                            lumen1[i, coord1_y, coord1_x] = 1
                        elif score1 < score2:
                            lumen2[i, coord1_y, coord1_x] = 1
                elif (lumen1[i + 1].max() > 0) ^ (
                    lumen2[i + 1].max() > 0
                ):  ####### 也可能上一层中只有一个腔有东西，发生了腔体压缩缺失/失去真腔形态的错分，这里要优化
                    if seg_regions.max() == 1:  ### 那么当：当前层也只有1个腔
                        coord1_y, coord1_x = np.where(seg_regions)
                        score1 = containing_score(
                            seg_regions, lumen1[i + 1]
                        )  ### 直接比直接分
                        score2 = containing_score(seg_regions, lumen2[i + 1])
                        if score1 > score2:
                            lumen1[i, coord1_y, coord1_x] = 1
                        else:
                            lumen2[i, coord1_y, coord1_x] = 1
                    elif seg_regions.max() == 2:  ### 当前层有2个腔的时候
                        if lumen1[i + 1].max():  ### 看看是哪一个腔有东西: 腔1有东西
                            score1 = containing_score(
                                lumen1[i + 1], seg_regions == 1
                            )  ### 决定把和上一层长得比较像的给它，和之前"包含"策略的理念是反过来的
                            score2 = containing_score(lumen1[i + 1], seg_regions == 2)
                            if score1 > score2:  ### 如果第一个区域长得和上一层像
                                lumen1[i] = seg_regions == 1
                                lumen2[i] = seg_regions == 2
                            else:
                                lumen1[i] = seg_regions == 2
                                lumen2[i] = seg_regions == 1
                        else:  ### 腔2有东西的话：
                            score1 = containing_score(lumen2[i + 1], seg_regions == 1)
                            score2 = containing_score(lumen2[i + 1], seg_regions == 2)
                            if score1 > score2:
                                lumen2[i] = seg_regions == 1
                                lumen1[i] = seg_regions == 2
                            else:
                                lumen2[i] = seg_regions == 2
                                lumen1[i] = seg_regions == 1
                    else:  ### 最糟糕的情况是，上一层错分/挤压只有一个域，当前层却有好多域
                        for num in range(
                            1, seg_regions.max() + 1
                        ):  ### 这里还没有想出好办法解决，先来个最糙的试试
                            coord1_y, coord1_x = np.where(seg_regions == num)
                            score1 = containing_score(seg_regions == num, lumen1[i + 1])
                            score2 = containing_score(seg_regions == num, lumen2[i + 1])
                            if score1 > score2:
                                lumen1[i, coord1_y, coord1_x] = 1
                            elif score1 < score2:
                                lumen2[i, coord1_y, coord1_x] = 1
                else:  ######### 也可能上面的一层什么都没有,我们就找上两层
                    if (
                        lumen1[i + 2].max() > 0 and lumen2[i + 2].max() > 0
                    ):  ### 需要确定之前的那一层是不是两个腔都有东西: 是的
                        if (
                            seg_regions.max() == 1
                        ):  ######### 如果当前层发现只有一个域，我们可以猜测有可能是错分造成或者挤压狭窄
                            score1 = containing_score(
                                lumen1[i + 2], seg_regions
                            )  ### 看看和上边那个域比较像，决定了去计算哪个的变化均值
                            score2 = containing_score(lumen2[i + 2], seg_regions)
                            diff_ave = 0  ### 准备比较前5层的变化程度，计算变化均值
                            if score1 > score2:
                                for j in range(3, 7):
                                    diff_ave += (
                                        np.abs(
                                            int(np.sum(lumen1[i + j]))
                                            - int(np.sum(lumen1[i + j - 1]))
                                        )
                                        / 5
                                    )
                            else:
                                for j in range(3, 7):
                                    diff_ave += (
                                        np.abs(
                                            int(np.sum(lumen2[i + j]))
                                            - int(np.sum(lumen2[i + j - 1]))
                                        )
                                        / 5
                                    )
                            diff = np.abs(
                                int(np.sum(seg_regions)) - int(np.sum(lumen1[i + 1]))
                            )  ### 再计算一下当前的层的差异
                            if diff > 10 * diff_ave:  ### 如果变化超过往次变化均值的20倍，说明是错分
                                target = image.copy()
                                changed_region1 = target * binary_erosion(
                                    lumen1[i + 2], iterations=1
                                )  ### 改变marker的使用，来源于相邻层
                                changed_region2 = target * binary_erosion(
                                    lumen2[i + 2], iterations=1
                                )
                                seg_regions = my_watershed(
                                    changed_region1, changed_region2, target
                                )
                        elif (
                            seg_regions.max() == 2
                        ):  ######### 经验证发现有的时候有两个腔也不是很好，不能用net2做marker
                            sim_seg1to1 = containing_score(
                                lumen1[i + 2], seg_regions == 1
                            )  ### 打算通过与上一层的相似度来解决问题
                            sim_seg1to2 = containing_score(
                                lumen2[i + 2], seg_regions == 1
                            )
                            sim_seg2to1 = containing_score(
                                lumen1[i + 2], seg_regions == 2
                            )
                            sim_seg2to2 = containing_score(
                                lumen2[i + 2], seg_regions == 2
                            )
                            if np.sum(seg_regions == 1) > np.sum(
                                seg_regions == 2
                            ):  ### 先比较一下两个域中哪个大哪个小
                                if (
                                    sim_seg1to1 >= sim_thr
                                    and sim_seg1to2 >= sim_thr
                                    and sim_seg2to1 < (1 - sim_thr)
                                    and sim_seg2to2 < (1 - sim_thr)
                                ):
                                    target = image.copy()  ### 若相似度大的大于0.5，小的小于0.5，就是错分
                                    changed_region1 = target * binary_erosion(
                                        lumen1[i + 2], iterations=1
                                    )  ### 改变marker的使用，来源于相邻层
                                    changed_region2 = target * binary_erosion(
                                        lumen2[i + 2], iterations=1
                                    )
                                    seg_regions = my_watershed(
                                        changed_region1, changed_region2, target
                                    )
                            elif np.sum(seg_regions == 1) < np.sum(seg_regions == 2):
                                if (
                                    sim_seg2to1 >= sim_thr
                                    and sim_seg2to2 >= sim_thr
                                    and sim_seg1to1 < (1 - sim_thr)
                                    and sim_seg1to2 < (1 - sim_thr)
                                ):
                                    target = image.copy()  ### 若相似度大的大于0.5，小的小于0.5，就是错分
                                    changed_region1 = target * binary_erosion(
                                        lumen1[i + 2], iterations=1
                                    )  ### 改变marker的使用，来源于相邻层
                                    changed_region2 = target * binary_erosion(
                                        lumen2[i + 2], iterations=1
                                    )
                                    seg_regions = my_watershed(
                                        changed_region1, changed_region2, target
                                    )
                        for num in range(
                            1, seg_regions.max() + 1
                        ):  ### 目前我们直接采用遍历里面的所有的小区域，可能要优化
                            coord1_y, coord1_x = np.where(seg_regions == num)
                            score1 = containing_score(seg_regions == num, lumen1[i + 2])
                            score2 = containing_score(seg_regions == num, lumen2[i + 2])
                            if score1 > score2:
                                lumen1[i, coord1_y, coord1_x] = 1
                            elif score1 < score2:
                                lumen2[i, coord1_y, coord1_x] = 1

    return lumen1, lumen2, suplim, start, z.min()


def resize_and_getTFL(
    result, result_AD, factor, crop_range
):  ### 输入128的Net2结果，和256的Net1的AD结果，不需要截断
    result = result.squeeze()
    result_256 = resize(result, resize_factor=factor, backsize=True)  #### 把128的TL变回256
    result_256[result_256 > 0] = 1
    result_T = np.zeros(result_AD.shape)
    result_T[crop_range[0] : crop_range[1]] = result_256  #### 把TL塞回到完整的结构中
    result_F = result_AD - result_T * result_AD  #### 减法出假腔
    result_T = result_AD - result_F  #### 减法出真腔
    return result_T, result_F


def post_processing(Network_TL, Network_FL, result_AD):
    ########### 输入神经网络预测得到的真腔和假腔，以及网络1分割的主动脉 ###########
    tempTL = Network_TL.copy()
    tempFL = Network_FL.copy()

    ##### IOU追踪初步分离降主，得到两个独立的腔并赋予语义信息 ######
    lumen1, lumen2, suplim, start, inflim = TFL_separation(result_AD, tempTL, tempFL)
    Network_TL_in_range, Network_FL_in_range = (
        np.zeros(tempTL.shape),
        np.zeros(tempFL.shape),
    )
    Network_TL_in_range[inflim : suplim + 1] = tempTL[inflim : suplim + 1]
    Network_FL_in_range[inflim : suplim + 1] = tempFL[inflim : suplim + 1]
    overlapTL_1 = np.sum(Network_TL_in_range * lumen1) / np.sum(
        lumen1
    )  ### 我觉得看真腔占比比较好：真腔在哪个管中占比比较大
    overlapTL_2 = np.sum(Network_TL_in_range * lumen2) / np.sum(lumen2)
    if overlapTL_1 > overlapTL_2:  ### 说明lumen1的TL占比比较大
        lumenT = lumen1
        lumenF = lumen2
    else:
        lumenT = lumen2
        lumenF = lumen1

    ###### 根据得到的带有TFL语义信息的修正后降主动脉，判断假腔是否延伸到髂动脉，强制处理错误的部分 ########
    AD_z, _, _ = np.where(result_AD)
    slices_regions = []
    for layer in range(AD_z.min(), AD_z.min() + 100):
        AD_slice_label = measure.label(result_AD[layer])
        AD_slice_regions = measure.regionprops(AD_slice_label)
        slices_regions.append(len(AD_slice_regions))
    slices_regions = np.array(slices_regions)
    slices_regions[slices_regions != 1] = 0
    proportion = np.sum(slices_regions) / slices_regions.shape[0]
    if proportion > 0.96:  ### 意思是如果前90层中有90%都只有一个域，就说明这个case的假腔只到腹部就结束了
        lumenT[AD_z.min() : AD_z.min() + 80][
            lumenF[AD_z.min() : AD_z.min() + 80] > 0
        ] = 1
        lumenF[AD_z.min() : AD_z.min() + 80] = 0
        for layer in range(AD_z.min() + 80, AD_z.max()):
            recent_slice = result_AD[layer].copy()  ### 用膜片来判断是否是到达了底部没有假腔的地方
            recent_slice_label = measure.label(recent_slice)
            recent_slice_regions = measure.regionprops(recent_slice_label)
            recent_close_slice = binary_closing(recent_slice, iterations=5)
            flap = recent_close_slice - recent_slice
            if len(recent_slice_regions) == 1 and np.sum(flap) < 10:
                lumenT[layer][lumenF[layer] > 0] = 1
                lumenF[layer] = 0

    ############ 向下追溯纠正降主动脉错分区域 #####################
    for layer in range(suplim, -1, -1):
        wrongT = lumenF[layer] * tempTL[layer]
        wrongF = lumenT[layer] * tempFL[layer]
        if np.sum(wrongT) != 0:
            wty, wtx = np.where(wrongT)
            tempFL[layer, wty, wtx] = 1
            tempTL[layer, wty, wtx] = 0
        if np.sum(wrongF) != 0:
            wfy, wfx = np.where(wrongF)
            tempTL[layer, wfy, wfx] = 1
            tempFL[layer, wfy, wfx] = 0

    Final_TL = tempTL.copy()
    Final_FL = tempFL.copy()
    return Final_TL, Final_FL, suplim, start
