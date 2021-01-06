import os

import SimpleITK as sitk
import numpy as np
import torch
from apex import amp
from torch import nn
from torch import optim
from torch.nn import DataParallel
from torch.nn.modules.batchnorm import _BatchNorm

from datasets.split_combine import SplitComb
from train import LRLoader
from utils import myPrint, Averager
from train import choose_top1_connected_component, dynamic_choose_topk_vessel_connected_component


from apex.parallel import convert_syncbn_model, SyncBatchNorm


class BatchFiller:
    def __init__(self, batch_size=0):
        self.bs = batch_size
        self.sample_queue = []
        self.name_queue = []
        self.shape_queue = []
        self.sequence_idx_queue = []

    def enqueue(self, sample: list, name: list, shape: list, sequence_idx: list):
        self.sample_queue.extend(sample)
        self.name_queue.extend(name)
        self.shape_queue.extend(shape)
        self.sequence_idx_queue.extend(sequence_idx)

    def isFull(self, mode="batch"):
        if mode == "batch":
            return len(self.sample_queue) >= self.bs
        else:
            if len(self.name_queue) == 0:
                return False
            return self.name_queue[0] != self.name_queue[-1]

    def dequeue(self, mode="batch"):
        if mode == "batch":
            n_pieces = self.bs
        else:
            n_pieces = 0
            ini_name = self.name_queue[0]
            for name in self.name_queue:
                if ini_name == name:
                    n_pieces += 1
                else:
                    break

        sample_pop = self.sample_queue[:n_pieces]
        name_pop = self.name_queue[:n_pieces]
        shape_pop = self.shape_queue[:n_pieces]
        sequence_idx_pop = self.sequence_idx_queue[:n_pieces]
        self.sample_queue = self.sample_queue[n_pieces:]
        self.name_queue = self.name_queue[n_pieces:]
        self.shape_queue = self.shape_queue[n_pieces:]
        self.sequence_idx_queue = self.sequence_idx_queue[n_pieces:]

        return sample_pop, name_pop, shape_pop, sequence_idx_pop


class Trainer:
    def __init__(self, warp, args, ioer, emlist):
        self.warp = warp
        self.args = args
        self.ioer = ioer
        self.emlist = emlist
        self.half = args["half"]
        self.dtype = torch.float
        self.splitcomb = SplitComb(args)
        if isinstance(self.warp, DataParallel):
            self.net = self.warp.module.net
        else:
            self.net = self.warp.net

        self.lrfun = LRLoader.load(args["train"]["lr_func"])
        self.lr_arg = args["train"]["lr_arg"]
        self.epoch = args["train"]["epoch"]
        self.optimizer, self.scheduler = self.__get_opimizer(
            args["train"]["start_epoch"] - 1
        )

        self.save_dir = os.path.join(
            args["output"]["result_dir"], args["output"]["save_dir"]
        )

        self.writer = sitk.ImageFileWriter()
        testdir = args["output"]["test_dir"]
        if testdir is None:
            self.testdir = os.path.join(self.save_dir, "testout")
        else:
            self.testdir = os.path.join(self.save_dir, testdir)
        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)
        if 'choose_top1_connect_region' in args['prepare']:
            self.choose_top1 = args['prepare']['choose_top1_connect_region']
        else:
            self.choose_top1 = False

        if 'choose_topk_vessel_connect_region' in args['prepare']:
            self.choose_topk= args['prepare']['choose_topk_vessel_connect_region']
        else:
            self.choose_topk = False

        self.printf = myPrint(os.path.join(self.save_dir, "log.txt"))

    def __get_opimizer(self, last_epoch=0):
        weight_decay = self.args["train"]["weight_decay"]
        if self.args['train']['optimizer'] == 'SGD':
            optimizer = optim.SGD(
                [{"params": self.net.parameters(), "initial_lr": 1}],
                lr=self.lrfun(self.lr_arg, last_epoch, self.epoch),
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif self.args['train']['optimizer'] == 'Adam':
            optimizer = optim.Adam([{"params": self.net.parameters(), "initial_lr": 1}], lr=self.lrfun(self.lr_arg, last_epoch, self.epoch))
        elif self.args['train']['optimizer'] == 'AdamW':
            optimizer = optim.AdamW([{"params": self.net.parameters(), "initial_lr": 1}], lr=self.lrfun(self.lr_arg, last_epoch, self.epoch), weight_decay=weight_decay)

        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: self.lrfun(self.lr_arg, x, self.epoch),
            last_epoch=last_epoch,
        )
        if self.half:
            self.net, optimizer = amp.initialize(self.net.float(), optimizer)

        return optimizer, scheduler

    def __writeLossLog(self, phase, epoch, meanloss, loss_list=[], em_list=[], lr=None):
        if phase == "Train":
            st = "{}, Epoch {}, lr {:.1e} | total loss: {:.4f} | ".format(
                phase.ljust(6), epoch, lr, meanloss,
            )
        else:
            st = "{}, Epoch {:d} | total loss: {:.4f} | ".format(
                phase.ljust(6), epoch, meanloss,
            )
        for i, l in enumerate(loss_list):
            st += "loss {:d}: {:.4f}  | ".format(i, l)

        for i, e in enumerate(em_list):
            st += "em {}: {:.4f}  | ".format(i, e)

        self.printf(st)

        return

    def train(self, epoch, dataloader):
        use_cuda = torch.cuda.is_available()
        self.net.train()

        for m in self.net.modules():  ### freeze是处理batchnorm的
            if isinstance(m, _BatchNorm):
                if self.args["train"]["freeze"]:
                    m.eval()

        loss_avg = Averager()
        lls_avg = Averager()

        for batch_idx, sample in enumerate(dataloader):

            sequence_idx = sample[0]
            data, target, name = sample[0], sample[1], sample[2]
            if use_cuda:
                data = data.cuda()
                target = target.cuda()

            if len(sample) > 3:
                args = sample[3:]
            else:
                args = None

            total_loss, loss_list = self.warp(
                data, target, False, args
            )  #### warp一般是并行，返回一组batch的损失
            loss_avg.update(total_loss.mean().detach().cpu().numpy())
            loss_list = tuple([l.cpu().numpy() for l in loss_list])
            info = "Finish training %d out of %d, " % (batch_idx + 1, len(dataloader))
            for lid, l in enumerate(loss_list):
                info += "loss %d: %.4f, " % (lid, float(np.mean(l)))
            print(info)
            lls_avg.update(loss_list)
            self.optimizer.zero_grad()
            loss_scalar = torch.mean(total_loss)
            if self.half:  # Automated Mix Precision
                with amp.scale_loss(loss_scalar, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_scalar.backward()
            torch.nn.utils.clip_grad_value_(self.net.parameters(), 2)
            self.optimizer.step()
        self.scheduler.step(epoch=epoch)
        self.__writeLossLog(
            "Train",
            epoch,
            meanloss=loss_avg.val(),
            loss_list=lls_avg.val(),
            lr=self.scheduler.get_lr()[0],
        )

    def validate(self, epoch, val_loader):
        # 训练时，每隔一定的epoch且epoch>15进行验证
        if not self.args["val"]:
            if epoch % self.args["output"]['save_frequency'] != 0 and epoch > 0:
                return
            if 0 < epoch < 15:
                return
        self.net.eval()

        loss_avg = Averager()
        lls_avg = Averager()
        em_avg = Averager()
        bs = self.args["train"]["batch_size"]
        ### 
        data_filler = BatchFiller(bs)
        target_filler = BatchFiller(bs)
        pred_filler = BatchFiller()
        full_target_filler = BatchFiller()

        val_results = []
        
        with torch.no_grad():
            pred_idx = 0
            use_cuda = torch.cuda.is_available()
            total_sample = len(val_loader)
            for sample_idx, sample in enumerate(val_loader):  ### 不打乱顺序，根据batchsize来输入

                data, target, name = sample[0], sample[1], sample[2]

                if len(sample) > 3:
                    sequence_idx = sample[3]
                else:
                    args = None
                    sequence_idx = 0

                data = data.squeeze(0)
                target = target.squeeze(0)
                data_pieces, split_position = self.splitcomb.split(data)
                target_pieces, split_position = self.splitcomb.split(target)

                data_filler.enqueue(
                    sample=list(data_pieces),
                    name=[name[0] for _ in range(data_pieces.shape[0])],
                    shape=[split_position for _ in range(data_pieces.shape[0])],
                    sequence_idx=[sequence_idx for _ in range(data_pieces.shape[0])],
                )
                target_filler.enqueue(
                    sample=list(target_pieces),
                    name=[name[0] for _ in range(data_pieces.shape[0])],
                    shape=[split_position for _ in range(data_pieces.shape[0])],
                    sequence_idx=[sequence_idx for _ in range(data_pieces.shape[0])],
                )

                full_target_filler.enqueue(
                    sample=[target], name=name, shape=[None], sequence_idx=[sequence_idx]
                )

                if sample_idx + 1 == total_sample:
                    pad_num = max(bs - len(data_filler.sample_queue) % bs, 1)
                    data_filler.enqueue(
                        sample=[
                            np.zeros_like(data_pieces[0, :]) for _ in range(pad_num)
                        ],
                        name=["padding" for _ in range(pad_num)],
                        shape=[None for _ in range(pad_num)],
                        sequence_idx=[sequence_idx for _ in range(pad_num)],
                    )
                    target_filler.enqueue(
                        sample=[
                            (np.zeros_like(target_pieces[0, :])) for _ in range(pad_num)
                        ],
                        name=["padding" for _ in range(pad_num)],
                        shape=[None for _ in range(pad_num)],
                        sequence_idx=[sequence_idx for _ in range(pad_num)],
                    )
                    full_target_filler.enqueue(
                        sample=[np.zeros_like(target)],
                        name=["padding"],
                        shape=[None],
                        sequence_idx=[sequence_idx],
                    )
                while data_filler.isFull(mode="batch"):

                    data_batch, name, shape, sequence_idx = data_filler.dequeue(mode="batch")
                    target_batch, name, shape, sequence_idx = target_filler.dequeue(mode="batch")

                    if use_cuda:
                        data_batch = torch.from_numpy(
                            np.stack(data_batch, axis=0)
                        ).cuda()
                        target_batch = torch.from_numpy(
                            np.stack(target_batch, axis=0)
                        ).cuda()

                    total_loss, loss_list, logits = self.warp(
                        data_batch, target_batch, True, sequence_idx
                    )

                    pred_filler.enqueue(sample=list(logits), name=name, shape=shape, sequence_idx=sequence_idx)
                    loss_avg.update(total_loss.mean().detach().cpu().numpy())
                    loss_list = tuple([l.cpu().numpy() for l in loss_list])
                    lls_avg.update(loss_list)

                while pred_filler.isFull(mode="sample"):

                    pred_full, _, shape, sequence_idx = pred_filler.dequeue(mode="sample")
                    target_full, name, _, sequence_idx = full_target_filler.dequeue(mode="sample")
                    pred_full = self.splitcomb.combine(pred_full, shape[0])
                    pred_full = choose_top1_connected_component(model_pred=pred_full, choose_top1=self.choose_top1)
                    # pred_full = dynamic_choose_topk_vessel_connected_component(model_pred=pred_full, choose_topk=self.choose_topk)

                    em_list = []
                    if self.emlist is not None:
                        for em_fun in self.emlist:
                            em_list.extend(em_fun(pred_full, target_full[0]))
                        em_list = tuple([l.cpu().squeeze().numpy() for l in em_list])
                        em_avg.update(em_list)

                    curr_case_nii_path = os.path.join(self.testdir, name[0]) + "_pred.nii.gz"
                    os.makedirs(os.path.dirname(curr_case_nii_path), exist_ok=True)
                    self.writer.SetFileName(curr_case_nii_path)
                    self.writer.Execute(
                        sitk.GetImageFromArray((pred_full.cpu().squeeze(0).numpy()).astype(np.uint8))
                    )

                    curr_case_npy_path = os.path.join(self.save_dir, 'val_out', '{}.npy'.format(name[0]))
                    os.makedirs(os.path.dirname(curr_case_npy_path), exist_ok=True)
                    np.save(
                        curr_case_npy_path,
                        pred_full.cpu().squeeze(0).numpy().astype(np.uint8)
                    )

                    info = "Finish validation %d out of %d, name %s, " % (
                        pred_idx + 1,
                        len(val_loader),
                        name[0],
                    )
                    pred_idx += 1
                    for lid, l in enumerate(em_list):
                        info += "em %d: %.4f, " % (lid, l)
                    print(info)
                    val_results.append(info)

        if not self.args["val"]:
            if epoch % self.args["output"]["save_frequency"] == 0:
                self.ioer.save_file(self.net, epoch, self.args, 0)
            else:
                return

        if self.emlist is not None:
            em_list = em_avg.val()
        self.__writeLossLog(
            "Val",
            epoch,
            meanloss=loss_avg.val(),
            loss_list=lls_avg.val(),
            em_list=em_list,
        )
        
        with open(os.path.join(self.save_dir, '{}_val.txt'.format(epoch)), 'a') as f_out:
            f_out.write('\n'.join(val_results) + '\n\n')


class Tester(nn.Module):
    def __init__(self, net, args):
        super().__init__()
        self.args = args
        self.net = net
        self.writer = sitk.ImageFileWriter()

        self.save_dir = os.path.join(
            args["output"]["result_dir"], args["output"]["save_dir"]
        )

        testdir = args["output"]["test_dir"]
        if testdir is None:
            self.testdir = os.path.join(self.save_dir, "testout")
        else:
            self.testdir = os.path.join(self.save_dir, testdir)
        if not os.path.exists(self.testdir):
            os.mkdir(self.testdir)

        if 'choose_top1_connect_region' in args['prepare']:
            self.choose_top1 = args['prepare']['choose_top1_connect_region']
        else:
            self.choose_top1 = False

        if 'choose_topk_vessel_connect_region' in args['prepare']:
            self.choose_topk = args['prepare']['choose_topk_vessel_connect_region']
        else:
            self.choose_topk = False

        self.splitcomb = SplitComb(args)

    def test(self, test_loader):
        self.net.eval()
        bs = self.args["train"]["batch_size"]
        data_filler = BatchFiller(bs)
        pred_filler = BatchFiller()

        total_sample = len(test_loader)
        with torch.no_grad():

            for sample_idx, sample in enumerate(test_loader):

                data, target, name = sample[0], sample[1], sample[2]

                if len(sample) > 3:
                    sequence_idx = sample[3]
                else:
                    args = None
                    sequence_idx = 0
                if len(data.shape) == 6:
                    data = data.squeeze(0).squeeze(0)
                else:
                    data = data.squeeze(0)
                data_pieces, split_position = self.splitcomb.split(data)

                data_filler.enqueue(
                    sample=list(data_pieces),
                    name=[name[0] for _ in range(data_pieces.shape[0])],
                    shape=[split_position for _ in range(data_pieces.shape[0])],
                    sequence_idx=[sequence_idx for _ in range(data_pieces.shape[0])],
                )

                if sample_idx + 1 == total_sample:
                    pad_num = max(bs - len(data_filler.sample_queue) % bs, 1)
                    data_filler.enqueue(
                        sample=[
                            np.zeros_like(data_pieces[0, :]) for _ in range(pad_num)
                        ],
                        name=["padding" for _ in range(pad_num)],
                        shape=[None for _ in range(pad_num)],
                        sequence_idx=[sequence_idx for _ in range(pad_num)],
                    )
                while data_filler.isFull(mode="batch"):
                    data_batch, name, shape, sequence_idx = data_filler.dequeue(mode="batch")
                    data_batch = torch.from_numpy(np.stack(data_batch, axis=0)).cuda()

                    logits = self.net(data_batch, sequence_idx)
                    pred_filler.enqueue(sample=list(logits), name=name, shape=shape, sequence_idx=sequence_idx)

                while pred_filler.isFull(mode="sample"):
                    pred_full, tmp_name, shape, sequence_idx = pred_filler.dequeue(mode="sample")

                    pred_full = self.splitcomb.combine(
                        pred_full, shape[0]
                    )  # shape: C * 3D

                    pred_full = choose_top1_connected_component(model_pred=pred_full, choose_top1=self.choose_top1).cpu().numpy().astype(np.int16)
                    # pred_full = dynamic_choose_topk_vessel_connected_component(model_pred=pred_full, choose_top1=self.choose_topk).cpu().numpy().astype(np.int16)

                    # if pred_full.shape[0] == 1:  # single channnel after sigmoid
                    #     pred_full = (pred_full > 0.5).squeeze().cpu().numpy().astype(np.int16)
                    # else:  # multiple channels after softmax
                    #     pred_full = pred_full.argmax(dim=0, keepdim=False).cpu().numpy().astype(np.int16)

                    savepath = os.path.join(self.testdir, tmp_name[0])
                    print(savepath)
                    np.save(savepath + ".npy", pred_full)

                    self.writer.SetFileName(savepath + ".nii.gz")
                    self.writer.Execute(
                        sitk.GetImageFromArray((pred_full).astype(np.uint8))
                    )
