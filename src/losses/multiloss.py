#-*- coding:utf-8 -*-
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
sys.path.append(os.path.join(os.path.dirname(__file__),'../configs'))
from config import cfg



class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.01):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class Focal_loss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=0, OHEM_percent=0.6, smooth_eps=0, class_num=2, size_average=True):
        super(Focal_loss, self). __init__()
        self.gamma = gamma
        self.alpha = alpha
        self.OHEM_percent = OHEM_percent
        self.smooth_eps = smooth_eps
        self.class_num = class_num
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, logits, target):
        # logits:[b,c,h,w] label:[b,c,h,w]
        pred = logits.softmax(dim=1)
        if pred.dim() > 2:
            pred = pred.view(pred.size(0),pred.size(1),-1)   # b,c,h,w => b,c,h*w
            pred = pred.transpose(1,2)                       # b,c,h*w => b,h*w,c
            pred = pred.contiguous().view(-1,pred.size(2))   # b,h*w,c => b*h*w,c
            target = target.argmax(dim=1)
            target = target.view(-1,1) # b*h*w,1

        if self.alpha:
            self.alpha = self.alpha.type_as(pred.data)
            alpha_t = self.alpha.gather(0, target.view(-1)) # b*h*w
            
        #pt = pred.gather(1, target.unsqueeze(1)) # b*h*w
        pt = pred.gather(dim=-1, index=target.unsqueeze(1))
        diff = (1-pt) ** self.gamma

        FL = -1  * diff * pt.log()
        #OHEM = FL.topk(k=int(self.OHEM_percent * FL.size(0)), dim=0)
        if self.smooth_eps > 0:
            K = self.class_num
            #lce = -1 * torch.sum(pred.log(), dim=1) / K
            lce = -pred.log().mean(dim=-1)
            loss = (1-self.smooth_eps) * FL + self.smooth_eps * lce
        
        if self.size_average: 
            return loss.mean() # or OHEM.mean()
        else: 
            return loss.sum() # + OHEM.sum()

class MultiLoss(nn.Module):
    """
    calculate the L2 and Num_loss
    """
    def __init__(self,alpha=0.25,gamma=1.0):
        super(MultiLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """Multi Loss
        Args:
            predictions (tuple): network output density map,
            targets (tensor): Ground truth  labels for a batch,
                shape: [batch_size,imgh,imgw] 
        """
        # pred_fg = predictions[:,1]
        # pred_bg = predictions[:,0]
        # alpha_factor_fg = torch.ones(targets.shape).cuda() * self.alpha
        # alpha_factor_bg =  1. - alpha_factor_fg
        # focal_weight_fg = alpha_factor_fg * torch.pow(1-pred_fg,self.gamma)
        # focal_weight_bg = alpha_factor_bg * torch.pow(1-pred_bg,self.gamma)
        # loss_c = -(focal_weight_fg * targets * torch.log(pred_fg) + focal_weight_bg * (1-targets)* torch.log(pred_bg))
        loss_c = self.loss(predictions, targets)
        # print(loss_l.size())
        # loss_c = -(targets * torch.log(pred_fg)+(1-targets)*torch.log(pred_bg))
        # loss_c = self.sigmoid_focal_loss(predictions,targets,self.gamma,self.alpha)
        return loss_c

    def sigmoid_focal_loss(self,logits, targets, gamma, alpha):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(num_classes, dtype=dtype, device=device).unsqueeze(0)
        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        term1 = torch.pow((1 - p), gamma) * torch.log(p)
        term2 = torch.pow(p ,gamma) * torch.log(1 - p)
        return -((t == class_range)*(t>=1)).float() * term1 * alpha - ((t == class_range) * (t == 0)).float() * term2 * (1 - alpha)

class CrossEntropyLossOneHot(nn.Module):
    def __init__(self):
        super(CrossEntropyLossOneHot, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, labels):
        return torch.mean(torch.sum(-labels * self.log_softmax(preds), -1))