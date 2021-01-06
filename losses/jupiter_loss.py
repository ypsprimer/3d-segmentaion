import torch
import torch.nn as nn
import torch.nn.functional as F


class Accuracy_Precision_Recall_IOU(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        """
        This score only fits BINARY classification.
        """
        assert outs.shape[0] == 1

        logits = (outs > 0.5).float()
        lab = labs.float()
        
        if self.ignore_index is not None:
            valid_mask = lab != self.ignore_index
            logits = logits.type(torch.cuda.FloatTensor)
            lab = lab.type(torch.cuda.FloatTensor)
            valid_mask = valid_mask.type(torch.cuda.FloatTensor)

            logits.mul_(valid_mask)  # inplace operation to save memory
            lab.mul_(valid_mask)

        tp_num = torch.sum((lab * logits) > 0).float()
        tn_num = torch.sum((lab + logits) == 0).float()
        pred_num = torch.sum(logits > 0).float()
        mask_num = torch.sum(lab > 0).float()
        n_element = 1
        for dim in labs.shape[1:]:
            n_element *= dim
        accuracy = (tp_num + tn_num) / n_element
        precision = tp_num / pred_num
        recall = tp_num / mask_num

        union = ((logits == 1) | (lab == 1)).sum().float()
        intersection = ((logits == 1) & (lab == 1)).sum().float()
        IOU = intersection / union

        return [accuracy.detach(), precision.detach(), recall.detach(), IOU.detach()]


class BinaryDiceScore(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        assert outs.shape[0] == 1

        labs = labs.bool().cuda().float()
        outs = (outs.cuda() > 0.5).float()

        loss = (2.0 * torch.sum(outs * labs) + 1e-4) / (
            torch.sum(outs) + torch.sum(labs) + 1e-4
        )

        return [loss.detach()]


class BinaryDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        assert logit.shape[1] == 1

        label = label.bool().float()

        loss = -(2.0 * torch.sum(logit * label) + 1e-4) / (
            torch.sum(logit) + torch.sum(label) + 1e-4
        )
        return loss, [loss.detach()]


class MultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        """
        :param logit: tensor@cuda [bs, class_num, z, y, x]
        :param label: tensor@cuda [bs, label_channel(default=1), z, y, x]
        :return:
        """
        num_classes = logit.size(1)
        assert num_classes > 1

        batch_size = logit.size(0)

        final_loss = 0
        for bs in range(batch_size):
            bs_loss = torch.tensor(0.).cuda()
            for i in range(1, num_classes):
                curr_prob = logit[bs, i, :, :, :].float()
                curr_label = (label[bs][0] == i).float()
                curr_loss = (2.0 * torch.sum(curr_label * curr_prob) + 1e-4) / (
                        torch.sum(curr_prob) + torch.sum(curr_label)
                )
                bs_loss += curr_loss
            final_loss += bs_loss / (num_classes - 1)
            
        final_loss = -final_loss / batch_size
        
        return final_loss, [final_loss.detach()]


class SectionSegMultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        """
        :param logit: tensor@cuda [bs, class_num, z, y, x]
        :param label: tensor@cuda [bs, label_channel(default=1), z, y, x]
        :return:
        """
        num_classes = logit.size(1)
        assert num_classes > 1

        batch_size = logit.size(0)

        final_loss = 0
        for bs in range(batch_size):
            liver_seg = (label[bs][0] > 0).float()
            bs_loss = torch.tensor(0.).cuda()
            for i in range(1, num_classes):
                curr_prob = logit[bs, i, :, :, :].float() * liver_seg
                curr_label = (label[bs][0] == i).float()
                curr_loss = (2.0 * torch.sum(curr_label * curr_prob) + 1e-4) / (
                        torch.sum(curr_prob) + torch.sum(curr_label) + 1e-4
                )
                bs_loss += curr_loss
            final_loss += bs_loss / (num_classes - 1)

        final_loss = -final_loss / batch_size

        return final_loss, [final_loss.detach()]


class EdgeSectionSegMultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=None, lamda=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lamda = lamda

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        """
        :param logit: tensor@cuda [bs, class_num, z, y, x]
        :param label: tensor@cuda [bs, label_channel(2), z, y, x]
        :return:
        """
        num_classes = logit.size(1)
        assert label.size(1) >= 2
        assert num_classes > 1

        batch_size = logit.size(0)

        regular_label = label.clone()[:, 0]
        edge_label = label.clone()[:, 1]

        final_loss = 0
        final_edge_loss = 0
        for bs in range(batch_size):
            liver_seg = (regular_label[bs] > 0).float()
            bs_loss = torch.tensor(0.).cuda()
            bs_edge_loss = torch.tensor(0.).cuda()
            curr_bs_edge = (edge_label[bs] > 0).float()
            for i in range(1, num_classes):
                curr_prob = logit[bs, i, :, :, :].float() * liver_seg
                curr_label = (regular_label[bs] == i).float()
                curr_loss = (2.0 * torch.sum(curr_label * curr_prob) + 1e-4) / (
                        torch.sum(curr_prob) + torch.sum(curr_label) + 1e-4
                )
                bs_loss += curr_loss

                curr_edge_loss = (2.0 * torch.sum(curr_label * curr_prob * curr_bs_edge) + 1e-4) / (
                        torch.sum(curr_prob * curr_bs_edge) + torch.sum(curr_label * curr_bs_edge)
                )
                bs_edge_loss += curr_edge_loss

            final_loss += bs_loss / (num_classes - 1)
            final_edge_loss += bs_edge_loss / (num_classes - 1)

        final_loss = -final_loss / batch_size
        final_edge_loss = -final_edge_loss / batch_size

        final_loss = final_loss + self.lamda * final_edge_loss

        return final_loss, [final_loss.detach()]

# now used
class OrganSegMultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        """
        ========================
        === Organ Seg专用Loss ===
        ========================
        遍历batch:
            1. 如果label.max == 1, 即当前的case为Liver Seg, 那么只考虑output中第2个channel的loss
            2. 如果label.max == 2, 即当前的case为Organ Seg, 那么会考虑output中第2、3个channel的loss
        :param logit: tensor@cuda --> [batch_size, class_num, z, y, x]
        :param label: tensor@cuda --> [batch_size, z, y, x] belongs to {0, 1}(liver) OR {0, 1, 2}(organ)
        :return:
        """
        output_channel_num = logit.size(1)
        assert output_channel_num == 3

        batch_size = logit.size(0)

        final_loss = 0
        for bs in range(batch_size):
            num_classes = round(label[bs].max().item()) + 1 # num_classes == 2(liver) OR 3(organ)
            bs_loss = torch.tensor(0.).cuda()
            for i in range(1, num_classes):
                curr_prob = logit[bs, i, :, :, :].float()
                curr_label = (label[bs] == i).float()
                curr_loss = (2.0 * torch.sum(curr_label * curr_prob) + 1e-4) / (
                        torch.sum(curr_prob) + torch.sum(curr_label)
                )
                bs_loss += curr_loss
            final_loss += (bs_loss / (num_classes - 1)) if num_classes > 1 else (bs_loss / num_classes)

        final_loss = 1-final_loss / batch_size

        return final_loss, [final_loss.detach()]
    
# now used 
class MultiClassDiceScore(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        assert outs.shape[0] == 1
        multi_class_dice_score_list = []
        avg_loss = 0
        class_num = round(labs.max().item())
        for i in range(1, class_num + 1):
            curr_labs = (labs == i).cuda().float()
            curr_outs = (outs == i).cuda().float()
            curr_loss = (2.0 * torch.sum(curr_outs * curr_labs) + 1e-4) / (torch.sum(curr_outs) + torch.sum(curr_labs) + 1e-4)
            multi_class_dice_score_list.append(curr_loss.detach())
            avg_loss += curr_loss.detach()
        
        multi_class_dice_score_list.append(avg_loss / class_num)

        return multi_class_dice_score_list


class SectionSegMultiClassDiceScore(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        assert outs.shape[0] == 1
        multi_class_dice_score_list = []
        avg_loss = 0
        class_num = round(labs.max().item())
        pred_result = outs[0].clone()
        liver_seg = (labs[0] > 0).cuda().float()
        for i in range(1, class_num + 1):
            curr_labs = (labs[0] == i).cuda().float()
            curr_outs = (pred_result == i).cuda().float() * liver_seg
            curr_loss = (2.0 * torch.sum(curr_outs * curr_labs) + 1e-4) / (
                        torch.sum(curr_outs) + torch.sum(curr_labs) + 1e-4)
            multi_class_dice_score_list.append(curr_loss.detach())
            avg_loss += curr_loss.detach()

        multi_class_dice_score_list.append(avg_loss / class_num)

        return multi_class_dice_score_list



class OrganSegMultiClassDiceScore(nn.Module):
    def __init__(self, ignore_index=None, class_num=3):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_num = class_num

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        """
        ========================
        === Organ Seg专用Score ===
        ========================
        遍历batch:
            1. 如果label.max == 1, 即当前的case为Liver Seg, 那么只考虑output中第2个channel的loss
            2. 如果label.max == 2, 即当前的case为Organ Seg, 那么会考虑output中第2、3个channel的loss
        :param logit: tensor@cuda --> [1, z, y, x] belongs to {0, 1, 2}
        :param label: tensor@cuda --> [1, z, y, x] belongs to {0, 1}(liver) OR {0, 1, 2}(organ)
        :return:
        """

        assert outs.shape[0] == 1
        multi_class_dice_score_list = []
        avg_loss = 0
        class_num = round(labs.max().item()) + 1 # class_num == 2(liver) OR 3(organ)
        for i in range(1, class_num):
            curr_labs = (labs == i).cuda().float()
            curr_outs = (outs == i).cuda().float()
            curr_loss = (2.0 * torch.sum(curr_outs * curr_labs) + 1e-4) / (
                        torch.sum(curr_outs) + torch.sum(curr_labs) + 1e-4)
            multi_class_dice_score_list.append(curr_loss.detach())
            avg_loss += curr_loss.detach()

        curr_score = (avg_loss / (class_num - 1)) if class_num > 1 else (avg_loss / class_num)
        multi_class_dice_score_list.append(curr_score)

        return multi_class_dice_score_list
    
    
class EdgeOrganSegMultiClassDiceLoss(nn.Module):
    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.lamda = 1.0

    def forward(self, logit: torch.Tensor, label: torch.Tensor) -> (torch.Tensor, list):
        """
        ========================
        === Organ Seg专用Loss ===
        ========================
        遍历batch:
            1. 如果label.max == 1, 即当前的case为Liver Seg, 那么只考虑output中第2个channel的loss
            2. 如果label.max == 2, 即当前的case为Organ Seg, 那么会考虑output中第2、3个channel的loss
        :param logit: tensor@cuda --> [batch_size, class_num, z, y, x]
        :param label: tensor@cuda --> [batch_size, lab_channel, z, y, x] belongs to {0, 1}(liver) OR {0, 1, 2}(organ)
        :return:
        """
        output_channel_num = logit.size(1)
        assert output_channel_num == 3
        assert label.size(1) == 2

        batch_size = logit.size(0)
        
        regular_label = label.clone()[:, 0]
        edge_label = label.clone()[:, 1]

        final_loss = 0
        final_edge_loss = 0
        for bs in range(batch_size):
            num_classes = round(regular_label[bs].max().item()) + 1 # num_classes == 2(liver) OR 3(organ)
            bs_loss = torch.tensor(0.).cuda()
            bs_edge_loss = torch.tensor(0.).cuda()
            curr_bs_edge = (edge_label[bs] > 0).float()
            for i in range(1, num_classes):
                curr_prob = logit[bs, i, :, :, :].float()
                curr_label = (regular_label[bs] == i).float()
                curr_loss = (2.0 * torch.sum(curr_label * curr_prob) + 1e-4) / (
                        torch.sum(curr_prob) + torch.sum(curr_label)
                )
                bs_loss += curr_loss
                
                curr_edge_loss = (2.0 * torch.sum(curr_label * curr_prob * curr_bs_edge) + 1e-4) / (
                        torch.sum(curr_prob * curr_bs_edge) + torch.sum(curr_label * curr_bs_edge)
                )
                bs_edge_loss += curr_edge_loss
                
            final_loss += (bs_loss / (num_classes - 1)) if num_classes > 1 else (bs_loss / num_classes)
            final_edge_loss += (bs_edge_loss / (num_classes - 1)) if num_classes > 1 else (bs_edge_loss / num_classes)

        final_loss = -final_loss / batch_size
        final_edge_loss = -final_edge_loss / batch_size
        
        final_loss = final_loss + self.lamda * final_edge_loss

        return final_loss, [final_loss.detach()]
    
    
class EdgeOrganSegMultiClassDiceScore(nn.Module):
    def __init__(self, ignore_index=None, class_num=3):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_num = class_num

    def forward(self, outs: torch.Tensor, labs: torch.Tensor) -> list:
        """
        ========================
        === Organ Seg专用Score ===
        ========================
        遍历batch:
            1. 如果label.max == 1, 即当前的case为Liver Seg, 那么只考虑output中第2个channel的loss
            2. 如果label.max == 2, 即当前的case为Organ Seg, 那么会考虑output中第2、3个channel的loss
        :param logit: tensor@cuda --> [1, z, y, x] belongs to {0, 1, 2}
        :param label: tensor@cuda --> [2, z, y, x] belongs to {0, 1}(liver) OR {0, 1, 2}(organ)
        :return:
        """

        assert outs.shape[0] == 1
        valid_lab = labs[:1]
        multi_class_dice_score_list = []
        avg_loss = 0
        class_num = round(valid_lab.max().item()) + 1 # class_num == 2(liver) OR 3(organ)
        for i in range(1, class_num):
            curr_labs = (valid_lab == i).cuda().float()
            curr_outs = (outs == i).cuda().float()
            curr_loss = (2.0 * torch.sum(curr_outs * curr_labs) + 1e-4) / (
                        torch.sum(curr_outs) + torch.sum(curr_labs) + 1e-4)
            multi_class_dice_score_list.append(curr_loss.detach())
            avg_loss += curr_loss.detach()

        curr_score = (avg_loss / (class_num - 1)) if class_num > 1 else (avg_loss / class_num)
        multi_class_dice_score_list.append(curr_score)

        return multi_class_dice_score_list