# -*- coding: utf-8 -*-
#
#  lr_scheme.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#
import numpy as np


def base_lr(learning_rate, epoch_i, epoch):
    if epoch_i <= epoch * 0.25:
        lr = learning_rate
    elif epoch_i <= epoch * 0.5:
        lr = 0.1 * learning_rate
    elif epoch_i <= epoch * 0.75:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    return lr


def constant(learning_rate, epoch_i, epoch):
    return learning_rate


def exponential_desend(initial_lr, current_epoch, total_epoch):
    slope = 2 / float(total_epoch - current_epoch)
    return pow(10, np.log10(initial_lr) - slope * current_epoch)


def base_lr_1(learning_rate, epoch_i, epoch):
    if epoch_i <= epoch * 0.5:
        lr = learning_rate
    elif epoch_i <= epoch * 0.75:
        lr = 0.1 * learning_rate
    elif epoch_i <= epoch * 0.90:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    return lr


def warm_up(learning_rate, epoch_i, epoch):
    if epoch_i <= 6:
        return learning_rate * epoch_i / 6
    elif epoch_i <= epoch * 0.5:
        lr = learning_rate
    elif epoch_i <= epoch * 0.75:
        lr = 0.1 * learning_rate
    elif epoch_i <= epoch * 0.90:
        lr = 0.01 * learning_rate
    else:
        lr = 0.001 * learning_rate
    return lr
