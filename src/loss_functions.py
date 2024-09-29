import torch
from torch import nn
import torch.nn.functional as F

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, **_):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, labels, **_):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = labels.view(-1)
        return self.bce_loss(probs_flat, targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0, **_):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, labels, **_):
        num = labels.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = labels.view(num, -1)
        intersection = (m1 * m2)

        # smooth = 1.

        score = 2. * (intersection.sum(1) + self.smooth) / (m1.sum(1) + m2.sum(1) + self.smooth)
        score = 1 - score.sum() / num
        return score

class BCEDiceLoss(nn.Module):
    def __init__(self, **_):
        super(BCEDiceLoss, self).__init__()
        self.bce = BCELoss2d()
        self.dice = SoftDiceLoss()

    def forward(self, logits, labels, **_):
        return self.bce(logits, labels) + self.dice(logits, labels)

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE