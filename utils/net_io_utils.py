#
#  net_io_utils.py
#  training
#
#  Created by AthenaX on 30/1/2018.
#  Copyright © 2018 Shukun. All rights reserved.
#
import os
import time

import torch

from collections import OrderedDict
from torch.nn import DataParallel
import shutil
from torch.jit import ScriptModule, script_method, trace
import json
import numpy as np
from models.inplace_abn import InPlaceABN, ABN
from datasets.split_combine import SplitComb
from apex import amp
from copy import deepcopy


def BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) or isinstance(module, ABN):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module


def apex_BN_convert_float(module):
    '''
    Designed to work with network_to_half.
    BatchNorm layers need parameters in single precision.
    Find all layers and convert them back to float. This can't
    be done with built in .apply as that function will apply
    fn to all modules, parameters, and buffers. Thus we wouldn't
    be able to guard the float conversion based on the module type.
    '''
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children():
        apex_BN_convert_float(child)
    return module


def BN_fuse(model):
    print('Begin to solve BN half problem!')
    for m in model.modules():
        if isinstance(m,ABN):
            mean = m.running_mean
            var = m.running_var
            w = m.weight
            b = m.bias

            newmean = torch.zeros_like(mean)
            newvar = torch.ones_like(var)
            neww = w/(torch.sqrt(var)+1e-4)
            newb = b-mean*w/(torch.sqrt(var) +1e-4)

            m.running_mean.data[:] = newmean
            m.running_var.data[:] = newvar
            m.weight.data[:] = neww
            m.bias.data[:] = newb
    return model
        # if torch.any(torch.isnan(newb)):
        #     print(neww,newb)


def my_load(net, state_dict, strict):
    keys = state_dict.keys()
    isparallel = all(["module" in k for k in keys])

    if isinstance(net, DataParallel):
        if isparallel:
            net.load_state_dict(state_dict, strict)
        else:
            net.module.load_state_dict(state_dict, strict)
    else:
        if isparallel:
            new_state_dict = OrderedDict()
            for k, v in net.items():
                name = k[7:]
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict, strict)
        else:
            net.load_state_dict(state_dict, strict)
    return net


class netIOer:
    def __init__(self, config):
        self.config = config
        self.save_dir = os.path.join(
            config["output"]["result_dir"], config["output"]["save_dir"]
        )
        self.save_freq = config["output"]["save_frequency"]
        self.strict = config["net"]["strict"]
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        if self.save_freq == 0:
            self.best_score = 10000  # generally, loss function, the lower the better
        self.split_comb = SplitComb(config)

    def load_file(self, net, net_weight_file):  # address the issue of DataParallel
        contents = torch.load(
            net_weight_file, map_location="cpu"
        )  # ('/home/data/dl_processor/net_params.pkl')
        state_dict = contents["state_dict"]
        net = my_load(net, state_dict, self.strict)

        if self.config["net"]["resume"] and self.config["train"]["start_epoch"] == 1:
            self.config["train"]["start_epoch"] = contents["epoch"]

        if self.save_freq == 0:
            if os.path.exists(os.path.join(self.save_dir, "best.pkl")):
                self.best_score = torch.load(os.path.join(self.save_dir, "best.pkl"))[
                    "loss"
                ]
            shutil(net_weight_file, os.path.join(self.save_dir, "starter.pkl"))

        # net.encoder_state_dict = contents["encoder_state_dict"]
        # net.encoder.load_state_dict(contents["encoder_state_dict"][0])
        return net, self.config


    def load_jit_file(self, net, jit_net_weight_file):  # address the issue of DataParallel
        contents = torch.jit.load(
            jit_net_weight_file, map_location="cpu"
        )  # ('/home/data/dl_processor/net_params.pkl')
        state_dict = contents.state_dict()
        net = my_load(net, state_dict, self.strict)

        # if self.config["net"]["resume"] and self.config["train"]["start_epoch"] == 1:
        #     self.config["train"]["start_epoch"] = contents["epoch"]

        if self.save_freq == 0:
            if os.path.exists(os.path.join(self.save_dir, "best.pkl")):
                self.best_score = torch.load(os.path.join(self.save_dir, "best.pkl"))[
                    "loss"
                ]
            shutil(jit_net_weight_file, os.path.join(self.save_dir, "starter.pkl"))

        # net.encoder_state_dict = contents["encoder_state_dict"]
        # net.encoder.load_state_dict(contents["encoder_state_dict"][0])
        return net, self.config

    def save_file(self, net, epoch, config, loss, isbreak=False):

        if isinstance(net, DataParallel):
            state_dict = net.module.state_dict()
            # encoder_state_dict = net.module.state_dict()
        else:
            state_dict = net.state_dict()
            # encoder_state_dict = net.state_dict()

        dicts = {
            # "encoder_state_dict": encoder_state_dict,
            # "decoder_state_dict": decoder_state_dict,
            "state_dict": state_dict,
            "epoch": epoch,
            "config": json.dumps(config),
            "loss": loss,
        }

        if isbreak:
            save_file = os.path.join(self.save_dir, "break.pkl")
            print("Manual interrupted, save to %s" % save_file)
            torch.save(dicts, save_file)
        if self.save_freq == 0:
            torch.save(dicts, os.path.join(self.save_dir, "last.pkl"))
            if loss < self.best_score:
                shutil.copy(
                    os.path.join(self.save_dir, "last.pkl"),
                    os.path.join(self.save_dir, "best.pkl"),
                )
                self.best_score = loss
                print("Replace old best.pkl, new best loss: %.4f" % loss)
        elif epoch % self.save_freq == 0 or epoch == self.config["train"]["epoch"]:
            torch.save(dicts, os.path.join(self.save_dir, "%03d.pkl" % epoch))
            # trace_model = self.trace(net)
            # trace_file = os.path.join(self.save_dir, "%03d.trace" % epoch)
            # torch.jit.save(trace_model, trace_file)

    def trace(self, model):
        config = self.config
        if config["net"]["load_weight"] == '':
            raise ValueError('you must load weight before jit')
        else:
            if config is not None:
                assert type(config) == dict
                with open(config["net"]["load_weight"].replace('.trace','_half.json').replace('.pkl','.json'), 'w') as f_out:
                    json.dump(config, f_out)
                print('save config to ', config["net"]["load_weight"].replace('.pkl','.json'))

            inference_batch = config["train"]["inference_batch_size"] if "inference_batch_size" in config["train"] else 1
            shape = config["prepare"]['crop_size']
            channel = config["prepare"]['channel_input']
            sample_data = torch.rand(inference_batch, channel, shape[0],shape[1],shape[2]).float().cuda()
            model = model.eval()
            if config["half"]:
                print('Model Half!')
                sample_data = sample_data.half()
            with torch.no_grad():
                trace = torch.jit.trace(model, sample_data)
                weight_file = config["net"]["load_weight"]
                trace_file = weight_file.replace('.trace','_half.trace')
                trace_file = trace_file.replace('.pkl','.trace')
                torch.jit.save(trace, trace_file)
                print('save model to ', trace_file)
                
    def compare_model_saver(self, model_fp32, model_fp16, val_data=None):

        def tmp_save_file(state_dict, epoch, config, save_name, save_dir=None, loss=0):

            dicts = {
                # "encoder_state_dict": encoder_state_dict,
                # "decoder_state_dict": decoder_state_dict,
                "state_dict": state_dict,
                "epoch": epoch,
                "config": json.dumps(config),
                "loss": loss,
            }
            if save_dir is not None:
                torch.save(dicts, os.path.join(save_dir, save_name))
            else:
                torch.save(dicts, save_name)

        config = self.config
        if config["net"]["load_weight"] == '':
            raise ValueError('you must load weight before save model!')
        else:
            pkl_files = torch.load(config["net"]["load_weight"], map_location="cpu")
            if type(pkl_files) == dict:
                curr_state_dict = pkl_files["state_dict"]
            else:
                curr_state_dict = pkl_files

            if val_data is not None:
                data, fullab, name = val_data[0]
                crop_img, zhw = self.split_comb.split(data)
                crop_img = crop_img.astype("float32")
                sample_data = torch.from_numpy(crop_img[len(crop_img)//3][np.newaxis]).float().cuda()
            else:
                inference_batch = config["train"]["inference_batch_size"] if "inference_batch_size" in config["train"] else 1
                shape = config["prepare"]['crop_size']
                channel = config["prepare"]['channel_input']
                sample_data = torch.ones(inference_batch, channel, shape[0], shape[1], shape[2]).float().cuda()
            model_fp32 = my_load(model_fp32, curr_state_dict, config["net"]["strict"])
            model_fp32 = model_fp32.eval()
            
            model_fp16 = my_load(model_fp16, curr_state_dict, config["net"]["strict"])
            model_fp16 = model_fp16.eval()
                
            with torch.no_grad():
                if val_data is not None:
                    pred_pieces = [
                        model_fp32(
                            torch.from_numpy(sub_crop_img[np.newaxis]).cuda()
                        ).cpu()
                        for sub_crop_img in crop_img
                    ]
                    pred_pieces = torch.cat(pred_pieces)

                    zhw = torch.from_numpy(zhw)
                    # comb_pred: [class_num, z, y, x]
                    fp32_output = self.split_comb.combine(pred_pieces, zhw)
                    # fp32_output = torch.argmax(comb_pred, dim=0, keepdim=False)

                    pred_pieces = [
                        model_fp16(
                            torch.from_numpy(sub_crop_img[np.newaxis]).half().cuda()
                        ).cpu()
                        for sub_crop_img in crop_img
                    ]
                    pred_pieces = torch.cat(pred_pieces)

                    # comb_pred: [class_num, z, y, x]
                    fp16_output = self.split_comb.combine(pred_pieces.float(), zhw)
                    # fp16_output = torch.argmax(comb_pred, dim=0, keepdim=False)
                    
                else:
                    fp32_output = model_fp32(sample_data).float()
                    fp16_output = model_fp16(sample_data.half()).float()
                

                if val_data is not None:
                    print(torch.unique(torch.argmax(fp32_output, dim=0, keepdim=False)))
                    fp32_argmax = torch.argmax(fp32_output, dim=0, keepdim=False)
                    fp16_argmax = torch.argmax(fp16_output, dim=0, keepdim=False)
                    half_max_argmax_diff = torch.max(torch.abs(fp32_argmax - fp16_argmax))
                    half_argmax_diff_num = torch.sum(fp32_argmax != fp16_argmax)
                    half_argmax_diff_rate = half_argmax_diff_num / torch.sum(fp32_argmax > 0)
                    print('Max Argmax Diff of Val Data between FP32 and FP16 is {}\nMax Argmax Diff Num of Val Data between FP32 and FP16 is {}\nMax Argmax Diff Rate of Val Data between FP32 and FP16 is {}'.format(half_max_argmax_diff, half_argmax_diff_num, half_argmax_diff_rate))
                    

                    np.save('./FP32_result.npy', torch.argmax(fp32_output, dim=0, keepdim=False))
                    np.save('./FP16_result.npy', torch.argmax(fp16_output, dim=0, keepdim=False))
                    np.save('./ori_img.npy', data)

                    if half_max_argmax_diff == 0 or half_argmax_diff_num < 100:
                        config['half'] = True
                        tmp_save_file(curr_state_dict, epoch=100, config=config, save_name=config["net"]["load_weight"].replace('.pkl', '_FP16.pkl'))
                        print('Save as FP16 !!!')
                    else:
                        config['half'] = False
                        tmp_save_file(curr_state_dict, epoch=100, config=config, save_name=config["net"]["load_weight"].replace('.pkl', '_FP32.pkl'))
                        print('Save as FP32 !!!')
                else:
                    half_max_loss = torch.max(torch.abs(fp32_output - fp16_output))
                    half_max_relative_loss = torch.max(torch.abs(fp32_output - fp16_output) / torch.abs(fp32_output))
                    print('Max Loss between FP32 and FP16 is {}\nMax Relative Loss between FP32 and FP16 is {}'.format(
                        half_max_loss, half_max_relative_loss))


class Inference(torch.nn.Module):
    def __init__(self, model, clip: list = None, normalize: list = None):
        """
        :param model: a pytorch model
        :param clip: a list, the first is lower and the second is higher boundary, default: None
        :param normalize: a list, the first is mean and the second is std, default: None
        """

        super().__init__()

        self.model = model
        self.clip = clip
        self.normalize = normalize

        if self.clip:
            self.lower_boundary = self.clip[0]
            self.higher_boundary = self.clip[1]

        if self.normalize:
            self.mean = self.normalize[0]
            self.std = self.normalize[1]

    def forward(self, data):
        data[data < self.lower_boundary] = self.lower_boundary
        data[data > self.higher_boundary] = self.higher_boundary

        data = (data - self.mean) / self.std

        return self.model(data)
