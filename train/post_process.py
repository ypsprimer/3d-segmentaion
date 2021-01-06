import morphtorch as MT
import torch


def choose_top1_connected_component(model_pred, choose_top1=True):
    """
    choose the 1st largest connect component
    :param model_pred: tensor@cuda --> [class_num, z, y, x]
    :return: tensor@cuda --> [1, z, y, x]
    """
    class_num = model_pred.size(0)
    if class_num == 1:
        if choose_top1:
            pred_top1 = MT.keep_topk_im((model_pred>0.5).squeeze(0), k=1)
            return pred_top1.unsqueeze(0)
        else:
            return model_pred > 0.5

    model_classify = torch.argmax(model_pred, dim=0, keepdim=False) # [z, y, x]

    if choose_top1:
        top1_keep = torch.zeros_like(model_classify).long().cuda()
        for i in range(1, class_num):
            curr_top1_keep = MT.keep_topk_im(model_classify == i, k=1)
            top1_keep += curr_top1_keep.long()
        model_classify = model_classify * top1_keep
    return model_classify.unsqueeze(0)


def dynamic_choose_topk_vessel_connected_component(model_pred, choose_topk=True):
    """
    Dynamic choose k-st largest connect component
    :param model_pred: tensor@cuda --> [class_num, z, y, x]
    :return: tensor@cuda --> [1, z, y, x] belongs to {0, 1, 2, 3}
    """
    model_cls = torch.argmax(model_pred, dim=0, keepdim=False)
    if choose_topk:

        hv_ivc_lab = MT.label((model_cls == 1) | (model_cls == 3), connectivity=3)
        hv_ivc_reg_list = MT.regionprops(hv_ivc_lab)
        hv_ivc_reg_list = sorted(hv_ivc_reg_list, key=lambda x: -x.area)
        hv_ivc_max_area = hv_ivc_reg_list[0].area
        hv_ivc_keep = (hv_ivc_lab == hv_ivc_reg_list[0].label).int()
        k = 1
        while k < 3:
            if hv_ivc_reg_list[k].area >= 0.1 * hv_ivc_max_area:
                hv_ivc_keep += (hv_ivc_lab == hv_ivc_reg_list[k].label).int()
                k += 1
            else:
                break
        pv_lab = MT.label(model_cls == 2, connectivity=3)
        pv_reg_list = MT.regionprops(pv_lab)
        pv_reg_list = sorted(pv_reg_list, key=lambda x: -x.area)
        pv_max_area = pv_reg_list[0].area
        pv_keep = (pv_lab == pv_reg_list[0].label).int()
        k = 1
        while k < 3:
            if pv_reg_list[k].area >= 0.1 * pv_max_area:
                pv_keep += (pv_lab == pv_reg_list[k].label).int()
                k += 1
            else:
                break
        model_cls = model_cls.int() * (hv_ivc_keep + pv_keep)
    return model_cls.unsqueeze(0)