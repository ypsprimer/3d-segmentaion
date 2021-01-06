import os

def get_split(file_path):

    root_dir = '/ssd/Jupiter/organ_seg/organ_seg_splits/'
    dwi_file = open(os.path.join(root_dir, '20201217_dwi_liver_train.txt'), 'w')
    t2_file = open(os.path.join(root_dir, '20201217_t2_liver_train.txt'), 'w')

    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            if 'DI' in line:
                line = line.strip()
                if 'T2' in line:
                    t2_file.write(line + '\n')
                elif 'DWI' in line:
                    dwi_file.write(line + '\n')
    
    dwi_file.close()
    t2_file.close()
    


if __name__ == '__main__':
    get_split('/ssd/Jupiter/organ_seg/organ_seg_splits/20201217_thick_liver_train.txt')