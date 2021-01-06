import numpy as np


def choose_roi_without_label(im, liver_seg_lab, config):
    """
    Function to choose Liver's RoI (for Ves Seg Model) based on liver_seg_lab
    1. Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    2. Dilate z,y,x of ROI depends on ZYX_ratio
    :param im: [z, y, x]
    :param liver_seg_lab: [z, y, x]
    :param config: 
    :return: roi_img: [1, z, y, x]
    """
    if "ROI_dilate_rate" in config["prepare"]:
        ROI_dilate_rate = float(config["prepare"]["ROI_dilate_rate"])
    else:
        ROI_dilate_rate = 0.1

    # Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    new_img_shape = np.shape(im)
    z_axis_seg_index, y_axis_seg_index, x_axis_seg_index = np.where(liver_seg_lab > 0)
    y_roi_lower, y_roi_higher = np.min(y_axis_seg_index), np.max(y_axis_seg_index)
    x_roi_lower, x_roi_higher = np.min(x_axis_seg_index), np.max(x_axis_seg_index)
    z_roi_lower, z_roi_higher = np.min(z_axis_seg_index), np.max(z_axis_seg_index)
    start_z, end_z, start_y, end_y, start_x, end_x = dilate_roi(
        z_roi_lower=z_roi_lower,
        z_roi_higher=z_roi_higher,
        y_roi_lower=y_roi_lower,
        y_roi_higher=y_roi_higher,
        x_roi_lower=x_roi_lower,
        x_roi_higher=x_roi_higher,
        img_shape=new_img_shape,
        ROI_dilate_rate=ROI_dilate_rate,
    )

    start_point_list = [start_z, start_y, start_x]
    curr_roi_info = {
        "start_point_list": start_point_list,
        "spacing_size": list(np.shape(im[0])),
    }
    final_img = im[start_z:end_z, start_y:end_y, start_x:end_x]

    return (
        final_img.astype(np.int16),
        curr_roi_info,
    )


def dilate_roi(
    z_roi_lower,
    z_roi_higher,
    y_roi_lower,
    y_roi_higher,
    x_roi_lower,
    x_roi_higher,
    img_shape,
    ROI_dilate_rate,
):
    z_roi_length = z_roi_higher + 1 - z_roi_lower
    y_roi_length = y_roi_higher + 1 - y_roi_lower
    x_roi_length = x_roi_higher + 1 - x_roi_lower

    z_high_limit = img_shape[0]
    y_high_limit = img_shape[1]
    x_high_limit = img_shape[2]

    z_left = max(z_roi_lower - z_roi_length * ROI_dilate_rate * 0.5, 0)
    z_right = min(
        z_roi_higher + z_roi_length * ROI_dilate_rate - (z_roi_lower - z_left),
        z_high_limit,
    )

    y_left = max(y_roi_lower - y_roi_length * ROI_dilate_rate * 0.5, 0)
    y_right = min(
        y_roi_higher + y_roi_length * ROI_dilate_rate - (y_roi_lower - y_left),
        y_high_limit,
    )

    x_left = max(x_roi_lower - x_roi_length * ROI_dilate_rate * 0.5, 0)
    x_right = min(
        x_roi_higher + x_roi_length * ROI_dilate_rate - (x_roi_lower - x_left),
        x_high_limit,
    )

    return (
        int(np.round(z_left)),
        int(np.round(z_right)),
        int(np.round(y_left)),
        int(np.round(y_right)),
        int(np.round(x_left)),
        int(np.round(x_right)),
    )


def choose_roi(im, label, liver_seg_lab, config):
    """
    Function to choose Liver's RoI (for Ves Seg Model) based on liver_seg_lab
    1. Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    2. Dilate z,y,x of ROI depends on ZYX_ratio
    :param im: [z, y, x]
    :param label: [lab_channel, z, y, x]
    :param liver_seg_lab: [z, y, x]
    :param config:
    :return: roi_img: [1, z, y, x]; roi_label: [lab_channel, z, y, x]
    """
    if "ROI_dilate_rate" in config["prepare"]:
        ROI_dilate_rate = float(config["prepare"]["ROI_dilate_rate"])
    else:
        ROI_dilate_rate = 0.1

    # Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    new_img_shape = np.shape(im)
    z_axis_seg_index, y_axis_seg_index, x_axis_seg_index = np.where(liver_seg_lab > 0)
    y_roi_lower, y_roi_higher = np.min(y_axis_seg_index), np.max(y_axis_seg_index)
    x_roi_lower, x_roi_higher = np.min(x_axis_seg_index), np.max(x_axis_seg_index)
    z_roi_lower, z_roi_higher = np.min(z_axis_seg_index), np.max(z_axis_seg_index)
    start_z, end_z, start_y, end_y, start_x, end_x = dilate_roi(
        z_roi_lower=z_roi_lower,
        z_roi_higher=z_roi_higher,
        y_roi_lower=y_roi_lower,
        y_roi_higher=y_roi_higher,
        x_roi_lower=x_roi_lower,
        x_roi_higher=x_roi_higher,
        img_shape=new_img_shape,
        ROI_dilate_rate=ROI_dilate_rate,
    )

    start_point_list = [start_z, start_y, start_x]
    curr_roi_info = {
        "start_point_list": start_point_list,
        "spacing_size": list(np.shape(im)),
    }
    final_img = im[start_z:end_z, start_y:end_y, start_x:end_x]
    final_label = label[:, start_z:end_z, start_y:end_y, start_x:end_x]

    return (
        final_img,
        final_label.astype(np.uint8),
        curr_roi_info,
    )


def choose_roi_multi_channel_img(im, label, liver_seg_lab, config):
    """
    Function to choose Liver's RoI (for Ves Seg Model) based on liver_seg_lab
    1. Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    2. Dilate z,y,x of ROI depends on ZYX_ratio
    :param im: [img_channel, z, y, x]
    :param label: [lab_channel, z, y, x]
    :param liver_seg_lab: [z, y, x]
    :param config:
    :return: roi_img: [1, z, y, x]; roi_label: [lab_channel, z, y, x]
    """
    if "ROI_dilate_rate" in config["prepare"]:
        ROI_dilate_rate = float(config["prepare"]["ROI_dilate_rate"])
    else:
        ROI_dilate_rate = 0.1

    # Get ROI depends on LiverSeg And Dilate by ROI_dilate_rate
    new_img_shape = np.shape(im)[1:]
    z_axis_seg_index, y_axis_seg_index, x_axis_seg_index = np.where(liver_seg_lab > 0)
    y_roi_lower, y_roi_higher = np.min(y_axis_seg_index), np.max(y_axis_seg_index)
    x_roi_lower, x_roi_higher = np.min(x_axis_seg_index), np.max(x_axis_seg_index)
    z_roi_lower, z_roi_higher = np.min(z_axis_seg_index), np.max(z_axis_seg_index)
    start_z, end_z, start_y, end_y, start_x, end_x = dilate_roi(
        z_roi_lower=z_roi_lower,
        z_roi_higher=z_roi_higher,
        y_roi_lower=y_roi_lower,
        y_roi_higher=y_roi_higher,
        x_roi_lower=x_roi_lower,
        x_roi_higher=x_roi_higher,
        img_shape=new_img_shape,
        ROI_dilate_rate=ROI_dilate_rate,
    )

    start_point_list = [start_z, start_y, start_x]
    curr_roi_info = {
        "start_point_list": start_point_list,
        "spacing_size": list(np.shape(im[0])),
    }
    final_img = im[:, start_z:end_z, start_y:end_y, start_x:end_x]
    final_label = label[:, start_z:end_z, start_y:end_y, start_x:end_x]

    return (
        final_img,
        final_label.astype(np.uint8),
        curr_roi_info,
    )