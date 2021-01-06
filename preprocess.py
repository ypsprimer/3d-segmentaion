"""
划分数据集，清洗标签

"""
import numpy as np 
import os
import random


def ids_split(all_ids, ratio=6):
    """
    :param all_ids: case 前缀 'DI_BM41647122'
    :param ratio: 划分比例，ratio = val/all

    return:
        train_ids: 
        val_ids: 
    """
    val_ids = set(random.sample(all_ids, len(all_ids)//ratio))
    train_ids = all_ids - val_ids
    print('train : val = {} : {} ({} : {})'.format(len(train_ids), len(val_ids), len(train_ids)//len(val_ids), 1))
    
    return train_ids, val_ids


def get_ids(data_dir):
    """
    获取数据id，目录下，去重
    :param data_dir
    
    return:
        id_set: 去重后的id
    """
    id_set = set()
    for i in os.listdir(data_dir):
        if 'DI' in i:
            name = 'DI_' + i.split('_')[1]
            if name not in id_set:
                id_set.add(name)
    
    return id_set


def write_txt(dir_name, ids_set, dst):
    """
    写入文件，...train.txt, ...val.txt
    :param dir_name: 当前目录
    :param ids_set: 
    :param dst: 输出文件路径
    """
    with open(dst, 'a') as f:
        for aid in ids_set:
            tol_name = os.path.join(dir_name, aid)
            # print(tol_name)
            f.write(tol_name + '\n')


def split_check(root_dir):
    """
    检查是否有交集，train & val
    检查是否train & val中的路径都存在
    
    """
    print('*' * 10)
    train_set = set()
    with open(liver_train_txt, 'r') as f:
        for line in f:
            # 路径存在
            assert os.path.exists(os.path.join(root_dir, line.strip() + '_new_raw.npy'))
            name = line.strip('\n').split('/')[-1]
            if name not in train_set:
                train_set.add(name)
    with open(organ_train_txt, 'r') as f:
        for line in f:
            assert os.path.exists(os.path.join(root_dir, line.strip() + '_new_raw.npy'))
            name = line.strip('\n').split('/')[-1]
            if name not in train_set:
                train_set.add(name)
    
    val_set = set()
    with open(liver_val_txt, 'r') as f:
        for line in f:
            assert os.path.exists(os.path.join(root_dir, line.strip() + '_new_raw.npy'))
            name = line.strip('\n').split('/')[-1]
            if name not in val_set:
                val_set.add(name)
    with open(organ_val_txt, 'r') as f:
        for line in f:
            assert os.path.exists(os.path.join(root_dir, line.strip() + '_new_raw.npy'))
            name = line.strip('\n').split('/')[-1]
            if name not in val_set:
                val_set.add(name)

    # 交集
    assert train_set & val_set == set()
            
    

if __name__ == '__main__':

    dwi_dir = '/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/LiverSeg_DWI_HB'
    t2_liver_dir = '/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/LiverSeg_FS_T2'
    t2_organ_dir = '/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/liver-spleen_data_FS_T2'

    liver_train_txt = '/ssd/Jupiter/organ_seg/organ_seg_splits/20201217_thick_liver_train.txt'
    organ_train_txt = '/ssd/Jupiter/organ_seg/organ_seg_splits/20201217_thick_organ_train.txt'

    liver_val_txt = '/ssd/Jupiter/organ_seg/organ_seg_splits/20201217_thick_liver_val.txt'
    organ_val_txt = '/ssd/Jupiter/organ_seg/organ_seg_splits/20201217_thick_organ_val.txt'

    dwi_ids = get_ids(data_dir=dwi_dir)
    t2_liver_ids = get_ids(t2_liver_dir)
    t2_organ_ids = get_ids(t2_organ_dir)
    print('Num of dwi cases: {}'.format(len(dwi_ids)))
    print('Num of t2 liver cases: {}'.format(len(t2_liver_ids)))
    print('Num of t2 organ cases: {}'.format(len(t2_organ_ids)))
    
    removed_set = t2_organ_ids - t2_liver_ids
    dwi_ids -= removed_set
    t2_liver_ids -= removed_set
    t2_organ_ids -= removed_set

    t2_only_liver_ids = t2_liver_ids - t2_organ_ids
    print('Num of t2 only liver cases:{}'.format(len(t2_only_liver_ids)))

    dwi_train, dwi_val = ids_split(dwi_ids)
    t2_onlyliver_train, t2_onlyliver_val = ids_split(t2_only_liver_ids)
    t2_organ_train, t2_organ_val = ids_split(t2_organ_ids)

    
    # dwi_train, t2_onlyliver_train, t2_organ_train -> OnlyLiver_trainset.txt
    # t2_organ_train -> Organ_trainset.txt
    # '''
    write_txt(dir_name=dwi_dir.split('/')[-1], ids_set=dwi_train, dst=liver_train_txt)
    write_txt(dir_name=t2_liver_dir.split('/')[-1], ids_set=t2_onlyliver_train, dst=liver_train_txt)
    write_txt(dir_name=t2_liver_dir.split('/')[-1], ids_set=t2_organ_train, dst=liver_train_txt)
    write_txt(dir_name=t2_organ_dir.split('/')[-1], ids_set=t2_organ_train, dst=organ_train_txt)

    # val ...
    write_txt(dir_name=dwi_dir.split('/')[-1], ids_set=dwi_val, dst=liver_val_txt)
    write_txt(dir_name=t2_liver_dir.split('/')[-1], ids_set=t2_onlyliver_val, dst=liver_val_txt)
    write_txt(dir_name=t2_liver_dir.split('/')[-1], ids_set=t2_organ_val, dst=liver_val_txt)
    write_txt(dir_name=t2_organ_dir.split('/')[-1], ids_set=t2_organ_val, dst=organ_val_txt)
    # '''

    split_check(root_dir='/ssd/Jupiter/organ_seg/OrganSeg_ThiCK_DynamicReshape/')





