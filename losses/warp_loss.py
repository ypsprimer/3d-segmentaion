import torch
from torch import nn


def neg(idx):
    if idx == 0:
        return None
    else:
        return -idx


class warpLoss(nn.Module):  # 这个函数是并行的
    def __init__(
        self, net, loss_fun, margin_training, margin_inference, emlist=None
    ):  
        """
        并行计算loss
        :param net: 
        :param loss_fun: [loss1, loss2, ..]
        :param margin_training: 训练时不计算loss的边缘大小 [z, y, x] -> [z:-z] ..
        :param margin_inference: 
        :param emlist: 评价指标 (not used)
        
        """
        super(warpLoss, self).__init__()
        self.net = net
        self.loss_funs = loss_fun
        self.emlist = emlist
        if margin_training:
            self.margin_training = margin_training
            self.margin_inference = margin_inference
        else:
            self.margin_training = margin_training
            self.margin_inference = margin_inference

    def forward(self, x, y, preds=False, *args):

        if args is not None:
            logit = self.net(x, args)
        else:
            logit = self.net(x)

        if preds:
            margin = self.margin_inference
        else:
            margin = self.margin_training

        valid_logit = logit[
            :,
            :,
            margin[0] : neg(margin[0]),
            margin[1] : neg(margin[1]),
            margin[2] : neg(margin[2]),
        ]
        valid_y = y[
            :,
            :,
            margin[0] : neg(margin[0]),
            margin[1] : neg(margin[1]),
            margin[2] : neg(margin[2]),
        ]

        loss_value = 0
        loss_list = []
        # 多个loss_funs
        if isinstance(self.loss_funs, list):
            for loss in self.loss_funs:
                output = loss(valid_logit, valid_y)
                # output[1] = output[0].detach()
                loss_value += output[0]
                loss_list.extend(output[1])
        else:
            output = self.loss_funs(valid_logit, valid_y)
            loss_value += output[0]
            loss_list.extend(output[1])

        if preds:
            return loss_value, loss_list, logit.detach()  # 用于产生预测值
        else:
            return loss_value, loss_list
