import numpy as np
import os
import json
import matplotlib.pyplot as plt


def file2matrix(file_name, output_path):
    train_loss = {}
    val_liver_loss = {}
    val_organ_loss = {}
    fr = open(file_name)
    # num_line = len(fr.readlines())         #get the number of lines in the file
    # print(num_line)
    # returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    # classLabelVector = []                       #prepare labels return   
    # fr = open(filename)
    # index = 0
    for line in fr:
        if line.startswith('Train'):
            line = line.strip().split('|')
            if len(line) == 4:
                seg_0 = line[0].split(',')
                epoch = seg_0[1].strip()[6:]
                lr = seg_0[2].strip()[3:]
                loss = line[1].strip()[12:]

                train_loss[int(epoch)] = [float(lr), float(loss)]


        elif line.startswith('Val'):
            line = line.strip().split('|')
            if len(line) == 6:
                seg_0 = line[0].split(',')
                epoch = seg_0[1].strip()[6:]
                loss = line[1].strip()[12:]
                em = line[3].strip()[6:]
                
                val_liver_loss[int(epoch)] = [float(loss), float(em)]
            
            elif len(line) == 7:
                # print(line)
                seg_0 = line[0].split(',')
                epoch = seg_0[1].strip()[6:]
                loss = line[1].strip()[12:]
                em_liver = line[3].strip()[6:]
                em_spline = line[4].strip()[6:]

                val_organ_loss[int(epoch)] = [float(loss), float(em_liver), float(em_spline)]
            # elif len(line) == 7:
                
    # print(train_loss)
    # print(val_organ_loss)
    # print(val_liver_loss)

    with open(os.path.join(output_path, 'train_loss.json'), 'w') as f:
        f.write(json.dumps(train_loss))

    with open(os.path.join(output_path, 'val_onlyliver_loss.json'), 'w') as f:
        f.write(json.dumps(val_liver_loss))
    
    with open(os.path.join(output_path, 'val_organ_loss.json'), 'w') as f:
        f.write(json.dumps(val_organ_loss))

    return train_loss, val_organ_loss, val_liver_loss


def get_plot(train, val_liver, val_organ): 

    epoch = sorted(train.keys())
    lr = []
    loss = []
    for i in epoch:
        lr.append(train[i][0])
        loss.append(train[i][1])

    val_epoch = sorted(val_liver.keys())
    val_loss = []
    em = []
    em_liver = []
    em_spline = []
    for i in val_epoch:
        val_loss.append(val_liver[i][0])
        em.append(val_liver[i][1])
        try:
            em_liver.append(val_organ[i][1])
            em_spline.append(val_organ[i][2])
        except:
            print(val_organ[i])
    
    
    assert len(val_loss) == len(em)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(epoch, loss, 'b', label = 'loss')
    # ax.legend(loc=0)
    ax.set_ylim(0,1)
    ax2 = ax.twinx()
    ax2.plot(val_epoch, em, '--r', label = 'em')
    ax2.plot(val_epoch, em_liver, '--g', label='em_liver')
    ax2.plot(val_epoch, em_spline, '--k', label='em_spline')
    ax2.set_ylim(0, 1)
    ax2.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax2.set_ylabel('dice')
    # ax2.legend(loc=0)

    plt.savefig('./xxx.png')



if __name__ == '__main__':
    log_path = '/yupeng/alg_jupiter_seg-local-organ_seg/results/xx2020_1219_onlyT2_adamw-decay/log.txt'
    # log_path = '/yupeng/alg_jupiter_seg-local-organ_seg/results/2020_1220_onlyT2_sgd-decay/log.txt'
    # log_path = '/yupeng/alg_jupiter_seg-local-organ_seg/results/2020_0730_UNET_ThiCK_OrganSeg_biggerCrop_dynamicThicknessNorm_base_v0_/log.txt'
    output_path = './loss_curve'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # dataMat, dataLabel = file2matrix(log_path)
    # print(dataMat, dataLabel)
    train_loss, val_organ_loss, val_liver_loss = file2matrix(log_path, output_path)
    get_plot(train_loss, val_liver_loss, val_organ_loss)
