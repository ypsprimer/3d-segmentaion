import argparse
import datetime
import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
import sys
import traceback
import warnings

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from configs import Config
from datasets import get_dataset, get_dataloader, check_multi_val
from losses import LossLoader, warpLoss
from models import ModelLoader
from trainer import Trainer, Tester
from utils import env_utils, netIOer
from models.bp_inplace_abn import ABN, InPlaceABN

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


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="ShuKun 3D segmentation model")
parser.add_argument(
    "--model",
    "-m",
    metavar="MODEL",
    default=None,
    help="model file to be used (default: sample)",
)
parser.add_argument(
    "--config", "-c", metavar="CONFIG", default="configs/test.yml", help="configs"
)
parser.add_argument(
    "-j",
    "--cpu_num",
    default=None,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "--epoch", default=None, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=None,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=None, type=int, metavar="N", help="mini-batch size "
)
parser.add_argument(
    "--gpu",
    type=str,
    default="0",
    metavar="N",
    help="the gpu used for training, \
                    separated by comma and no space left(default: 0)",
)
parser.add_argument(
    "--lr",
    default=None,
    type=float,
    metavar="LR",
    help="Learning rate, if specified, \
                    the default lr shall be replaced with this one",
)
parser.add_argument("--loss", default=None, type=str, help="the loss function used")
parser.add_argument(
    "--load-weight",
    default=None,
    type=str,
    metavar="PATH",
    help="path to loaded checkpoint, start from 0 epoch (default: none)",
)
parser.add_argument(
    "--resume",
    default=None,
    type=str,
    metavar="PATH",
    help="path to loaded checkpoint, start from that epoch (default: none)",
)
parser.add_argument(
    "--save-dir",
    default=None,
    type=str,
    metavar="SAVE",
    help="directory to store folders (default: none)",
)
parser.add_argument(
    "--test-dir",
    default=None,
    type=str,
    metavar="SAVE",
    help="directory to save test results",
)
parser.add_argument(
    "--save-frequency",
    default=None,
    type=int,
    metavar="SAVE",
    help="frequency of store snapshots (default: none)",
)
parser.add_argument(
    "--folder-label",
    default=None,
    type=str,
    metavar="SAVE",
    help="directory to save checkpoint (default: none)",
)
parser.add_argument(
    "--optimizer",
    default=None,
    type=str,
    metavar="O",
    help="optimizer used (default: sgd)",
)
parser.add_argument(
    "--cudnn",
    default=None,
    type=lambda x: (str(x).lower() in ("yes", "true", "t", "1")),
    help="cudnn benchmark mode",
)
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument("--freeze", action="store_true", help="freeze bn")
parser.add_argument("--test", action="store_true", help="test mode")
parser.add_argument("--val", action="store_true", help="val mode")
parser.add_argument('--jit', action = 'store_true', help='convert weight to jit model')
parser.add_argument('--jitval', action = 'store_true', help='use jit to lode model for val')
parser.add_argument('--jit2jithalf', action = 'store_true', help='use jit to lode model for val')
parser.add_argument('--savemodel', action = 'store_true', help='choose apex opt_level for model inference')


def prepare(config):
    env_utils.setEnvironment(config)
    model = ModelLoader.load(config["net"]["model"])
    if config["jit"]:
        model = ModelLoader.load(config["net"]["model"], abn=0)
    if config["debug"]:
        model = ModelLoader.load(config["net"]["model"], abn=1)
    loss = LossLoader.load(config["net"]["loss"], config)
    em = LossLoader.load(config["net"]["em"], config)
    netio = netIOer(config)

    if config["jit"]:
        if config["half"]:
            model = model.half()
            model = BN_convert_float(model)
        if config["net"]["load_weight"] != "":
            if config["jit2jithalf"]:
                model, config = netio.load_jit_file(model, config["net"]["load_weight"])
            else:
                model, config = netio.load_file(model, config["net"]["load_weight"])
        model = model.cuda()
        netio.trace(model)
        sys.exit()

    if config["savemodel"]:
        model_fp32 = ModelLoader.load(config["net"]["model"], abn=1)
        model_fp16 = ModelLoader.load(config["net"]["model"], abn=0)
        model_fp32 = model_fp32.cuda()
        model_fp16 = model_fp16.cuda()
        model_fp16 = model_fp16.half()
        val_data = get_dataset(config=config, phase='val')
        netio.compare_model_saver(model_fp32=model_fp32, model_fp16=model_fp16, val_data=val_data)
        sys.exit()

    if config["net"]["load_weight"] != "":
        if config["jitval"]:
            model, config = netio.load_jit_file(model, config["net"]["load_weight"])
        else:
            model, config = netio.load_file(model, config["net"]["load_weight"])
    # optimizer = optim.SGD(model.parameters(), lr= config.train['lr_arg'], momentum=0.9,
    #                       weight_decay=config.train['weight_decay'])

    model = model.cuda()

    if isinstance(loss, list):
        for loss_index in range(len(loss)):
            loss[loss_index] = loss[loss_index].cuda()
    else:
        loss = loss.cuda()

    if (
        "margin_training" in config["prepare"]
        and "margin_inference" in config["prepare"]
    ):  
        warp = warpLoss(
            model,
            loss,
            config["prepare"]["margin_training"],
            config["prepare"]["margin_inference"],
        )
    elif "margin" in config["prepare"]:
        warp = warpLoss(model, loss, config["prepare"]["margin"], config["prepare"]["margin"])
    else:
        warp = warpLoss(model, loss, 0)
    if not config["debug"]:
        warp = DataParallel(warp)

    trainer = Trainer(warp, config, netio, emlist=em)
    train_data = get_dataset(config, "train")  ### 这个dataset继承了pytorch的dataset
    # 不同的val_split
    multi_val, multi_val_num = check_multi_val(config)
    
    print(config["augtype"])  ### 分割运行时唯一输出的：增强类型
    train_loader = get_dataloader(
        train_data,
        batch_size=config["train"]["batch_size"],  ### train是1批1批出来的
        shuffle=True,
        num_workers=config["env"]["cpu_num"],
        drop_last=True,
        pin_memory=True,
    )

    if multi_val:
        val_data = [get_dataset(config, 'val', True, val_index) for val_index in range(multi_val_num)]
        val_loader = [get_dataloader(v, batch_size=1, shuffle=False, num_workers=config["env"]["cpu_num"], pin_memory=False)
                      for v in val_data]
    else:
        val_data = get_dataset(config, "val")
        val_loader = get_dataloader(
            val_data,
            batch_size=1,  ### val是1个1个出来的
            shuffle=False,
            num_workers=config["env"]["cpu_num"],
            pin_memory=False
        )
    return (
        config,
        model,
        loss,
        warp,
        trainer,
        train_data,
        val_data,
        train_loader,
        val_loader,
    )


def run(config):

    (
        config,
        model,
        loss,
        warp,
        trainer,
        train_data,
        val_data,
        train_loader,
        val_loader,
    ) = prepare(config)
    
    if config["test"]:
        print("Start testing")
        # if hasattr(model, 'test'):
        #    model.forward = model.test
        model = DataParallel(model.cuda())
        tester = Tester(model, config)
        val_data = get_dataset(config, "test")
        test_loader = get_dataloader(
            val_data,
            batch_size=1,
            shuffle=False,
            num_workers=3,
            pin_memory=False
        )
        tester.test(test_loader)
        return
    elif config["val"]:
        print("Start Val")
        start_epoch = config["train"]["start_epoch"]
        if isinstance(val_loader, list):
            for val_idx in range(len(val_loader)):
                trainer.validate(start_epoch, val_loader[val_idx])
        else:
            trainer.validate(start_epoch, val_loader)
    else:
        start_epoch = config["train"]["start_epoch"]
        epoch = config["train"]["epoch"]
        print("Start training from %d-th epoch" % start_epoch)

        # if isinstance(val_loader, list):
        #     for i in range(len(val_loader)):
        #         trainer.validate(0, val_loader[i])
        # else:
        #     trainer.validate(0, val_loader)
        for i in range(start_epoch, epoch + 1):
            try:
                trainer.train(i, train_loader)
                if isinstance(val_loader, list):
                    for val_idx in range(len(val_loader)):
                        trainer.validate(i, val_loader[val_idx])
                else:
                    trainer.validate(i, val_loader)
            except KeyboardInterrupt as e:
                traceback.print_exc()
                trainer.ioer.save_file(trainer.net, i, trainer.args, 1e10, isbreak=True)
                sys.exit(0)


def syncKeys(config, args):
    keys = {
        "net": ["model", "load_weight", "loss", "optimizer"],
        "train": ["start_epoch", "epoch", "batch_size", "freeze", "cudnn"],
        "output": ["save_frequency", "test_dir"],
    }
    for prop in keys:

        for k in keys[prop]:
            if prop in config:
                if getattr(args, k):
                    config[prop][k] = getattr(args, k)

    if args.resume is not None:
        config["net"]["load_weight"] = args.resume
        config["net"]["resume"] = True

    if (
        args.lr is not None
    ):  # override the yml setting, use constant lr, useful for hand tuning
        config["train"]["lr_arg"] = args.lr
        config["train"]["lr_func"] = "constant"

    if args.debug:
        config["output"]["save_dir"] = "tmp"
    else:
        if config["output"]["save_dir"] == 0:
            date = datetime.datetime.now()
            date = date.strftime("%Y%m%d")
            config["output"]["save_dir"] = date + "_" + config["net"]["model"]
    if args.save_dir is not None:
        config["output"]["save_dir"] = args.save_dir

    config["debug"] = args.debug
    config["test"] = args.test
    config["val"] = args.val
    config["jit"] = args.jit
    config["jitval"] = args.jitval
    config["jit2jithalf"] = args.jit2jithalf
    config["savemodel"] = args.savemodel
    return config


if __name__ == "__main__":

    global args
    args = parser.parse_args()
    # 0-7
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = Config.load(args.config)
    config = syncKeys(config, args)
    if config["train"]["cudnn"] and (not config["debug"]):
        torch.backends.cudnn.benchmark = True
        print("cudnn mode")

    else:
        torch.backends.cudnn.benchmark = False
        print("no cudnn mode")
    print(config["net"])
    run(config)

    # import time
    # t1 = time.time()
    # model = ModelLoader().load("unet_shared_encoders_medium")
    # # model.load_state_dict(torch.load("/Jupiter/workspaces/lx/results/0630/baseline/20200707_unet_shared_encoders_medium/break.pkl", map_location="cpu"))
    # model = model.cuda()
    # print(time.time() - t1)