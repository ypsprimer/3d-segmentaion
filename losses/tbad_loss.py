import torch.nn as nn
import torch
from torch.nn import functional as F
from morphtorch import dilate, erode


class dice_ce_mix_loss(nn.Module):
    def __init__(self):
        super(dice_ce_mix_loss, self).__init__()

    def forward(self, output, mask):
        num_classes = output.size(1)
        loss = 0
        loss_list = []
        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :, :]).float()
            mask_i = torch.squeeze(mask == i).float()

            if i in [0, 3]:
                mask_i += (probs * torch.squeeze(mask == 4).float()).squeeze().float()

            # Anyway, up to here, it will be a 3-d array

            num = torch.sum(probs * mask_i, dim=(-1, -2, -3))
            den1 = torch.sum(probs * probs, dim=(-1, -2, -3))
            den2 = torch.sum(mask_i * mask_i, dim=(-1, -2, -3))

            eps = 0.0000001
            dice = (-2 * num + eps) / (den1 + den2 + eps)
            loss += dice
            loss_list.append(dice.detach())

        return loss / num_classes, tuple(loss_list)


class MultiClassDiceScorewithBG(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, mask):
        output = output.cuda()
        mask = mask.cuda()
        num_classes = int(torch.max(mask))

        score_list = []
        output = torch.argmax(output, dim=0)
        for i in range(num_classes):

            probs_i = (output == i).float()
            mask_i = (mask == i).float()

            if i in [0, 3]:
                mask_i = (
                    mask_i
                    + ((probs_i == 1).float() * (mask == 4).float()).squeeze().float()
                )

            num = torch.sum(probs_i * mask_i, dim=(-1, -2, -3))
            den1 = torch.sum(probs_i * probs_i, dim=(-1, -2, -3))
            den2 = torch.sum(mask_i * mask_i, dim=(-1, -2, -3))

            eps = 0.0000001
            dice = (2 * num + eps) / (den1 + den2 + eps)

            score_list.append(dice)

        return score_list


class GAN_loss(nn.Module):
    def __init__(self):
        super(GAN_loss, self).__init__()
        self.CELoss = nn.BCELoss()

    def forward(self, output: tuple, mask: torch.Tensor) -> tuple:
        """
        :param output: tuple([pred, dis_pred, dis_gt])
        :param mask: same shape with pred
        :return:
        """
        mask = mask.long()
        g_segmentation_loss, dice_loss_list = dice_loss(output[0], mask)
        g_adversarial_loss = self.CELoss(
            output[1].float(), torch.ones_like(output[1], dtype=torch.float)
        )

        d_adversarial_fake_loss = self.CELoss(
            output[1].float(), torch.zeros_like(output[1], dtype=torch.float)
        )
        d_adversarial_real_loss = self.CELoss(
            output[2].float(), torch.ones_like(output[2], dtype=torch.float)
        )

        total_loss = (
            g_segmentation_loss
            + g_adversarial_loss
            + d_adversarial_real_loss
            + d_adversarial_fake_loss
        )
        loss_list = dice_loss_list + tuple(
            [
                g_adversarial_loss.detach(),
                d_adversarial_fake_loss.detach(),
                d_adversarial_real_loss.detach(),
            ]
        )

        return total_loss, loss_list


def dice_loss(x: torch.Tensor, mask: torch.Tensor):
    """
    :param x: 5D tensor (B - C - 3D)
    :param mask: 5D tensor (B - C - 3D) ONEHOT tensor or 3D tensor with integer labels
    :return: scaler
    """

    y = torch.zeros_like(x).scatter(1, mask.long(), value=1).float()

    num = (x * y).sum(dim=(-1, -2, -3))
    den1 = x.pow(2).sum(dim=(-1, -2, -3))
    den2 = y.pow(2).sum(dim=(-1, -2, -3))

    dice = (-2 * num + 1e-6) / (den1 + den2 + 1e-6)
    total_dice = dice.sum()
    loss_list = tuple(dice.T)
    loss_list = tuple([loss.detach() for loss in loss_list])
    return total_dice, loss_list
