from torch import nn
import torch.nn.functional as F
import torch


class SkeLossMultiClass(nn.Module):

    def __init__(self, ignore_index=255, beta=25.):
        super().__init__()
        self.ignore_index = ignore_index
        self.beta = beta

    def forward(self, output, labels):
        num_classes = output.size(1)
        assert num_classes > 2

        # ves, ske: [bs, z, y, x]
        ves = labels.clone()[:, 0]
        ske = labels.clone()[:, 1]

        batch_size = output.size(0)

        final_loss = 0
        for bs in range(batch_size):
            ske_eso = 0
            for i in range(1, num_classes):
                probs = output[bs, i, :, :, :].float()
                ves_i = (ves[bs] == i).float()
                ske_i = (ske[bs] == i).float()

                importance = torch.ones_like(ske_i) - ves_i + ske_i
                importance = importance.float() + 0.1
                importance[ves_i == self.ignore_index] = 0

                num = probs * ves_i * importance
                num = torch.sum(num)

                den1 = probs * importance
                den1 = torch.sum(den1)

                den2 = ves_i * importance
                den2 = torch.sum(den2)

                eps = 1e-4

                curr_ske_score = (((1. + self.beta) * num + eps) / (den1 + den2 * self.beta + eps))
                ske_eso += curr_ske_score
            final_loss += ske_eso / (num_classes - 1)

        final_loss = -final_loss / batch_size
        return final_loss, [final_loss.detach()]


class VesselSegMultiClassDiceScore(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        assert outs.shape[0] == 1
        multi_class_dice_score_list = []
        avg_loss = 0
        class_num = round(labs[0][labs[0]!=self.ignore_index].max().item())
        pred_result = outs[0].clone()
        for i in range(1, class_num + 1):
            curr_labs = (labs[0] == i).cuda().float()
            curr_outs = (pred_result == i).cuda().float()
            curr_loss = (2.0 * torch.sum(curr_outs * curr_labs) + 1e-4) / (
                        torch.sum(curr_outs) + torch.sum(curr_labs) + 1e-4)
            multi_class_dice_score_list.append(curr_loss.detach())
            avg_loss += curr_loss.detach()

        multi_class_dice_score_list.append(avg_loss / class_num)

        return multi_class_dice_score_list



class Ske_loss(nn.Module):
    def __init__(self, ignore_index, beta=25.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.diffloss = nn.MSELoss()
        self.beta = beta

        # importance: how many false positive you can afford to exchange for this true positive
        # self.beta: higher the value, higher the recall

    def forward(self, logit, labels):
        assert logit.shape[1] <= 2
        if logit.shape[1] == 1:
            outs = F.sigmoid(logit)
        else:
            outs = F.softmax(logit, dim=1)[:, 1:]

        ske = labels[:, 0:1]
        ves = labels[:, 1:2]
        importance = torch.ones_like(ske) - ves + ske
        importance = importance.float() + 0.1
        importance[ves == self.ignore_index] = 0
        ves = ves.float()
        # importance = (importance-1)*0.2+1

        tp = torch.sum(outs * ves * importance)
        nOut = torch.sum(outs * importance)
        nLab = torch.sum(ves * importance)

        loss = -((1.0 + self.beta) * tp + 1e-4) / (nOut + nLab * self.beta + 1e-4)
        #         print(['term1', torch.sum(outs * labels)])
        #         print(['weight_term1', torch.sum(outs * labels * importance)])
        #         print(['term2', torch.sum(outs)])
        #         print(['weight_term2', torch.sum(outs * importance)])
        #         print(['term3', torch.sum(labels)])
        #         print(['weight_term3', torch.sum(labels * importance)])
        #         print('recall %.4f, precision %.4f' %((( tp + 1e-4) / ( nLab + 1e-4)).detach().cpu().numpy(),
        #                                                (( tp + 1e-4) / ( nOut + 1e-4)).detach().cpu().numpy()))

        return loss, [loss.detach()]


class ske_recall(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        labels = labels[0]
        validmask = labels != self.ignore_index
        labels = labels * validmask
        tp = torch.sum(pred * labels)
        N_pos = torch.sum(labels)
        sample_loss = tp.float() / N_pos.float()
        return sample_loss


class ves_precision(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        labels = labels[1]
        validmask = labels != self.ignore_index
        labels = labels * validmask
        tp = torch.sum(pred * labels)
        N_pred = torch.sum(pred * validmask)
        sample_loss = tp.float() / N_pred.float()
        return sample_loss


class max_lost(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, labels):
        skes = (labels > 0) & (labels != self.ignore_index)
        lost_ske = skes * (1 - pred)
        all_lost = labels[lost_ske]
        if len(all_lost) == 0:
            return torch.zeros(1)[0]
        else:
            return torch.max(all_lost)
