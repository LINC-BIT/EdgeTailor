from abc import ABC, abstractmethod
import copy


class ABAlgModelsManager(ABC):
  # @abstractmethod
  # def forward_to_compute_task_loss(self, models, x, y):
#   raise NotImplementedError
    
    @abstractmethod
    def forward(self, models, x):
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, models, x):
        raise NotImplementedError
    
    @abstractmethod
    def get_accuracy(self, models, test_dataloader):
        raise NotImplementedError

    def get_model(self, models, key):
        model = models[key]
        if isinstance(model, tuple):
            model = model[0]
        return model
    
    def set_model(self, models, key, model):
        if key in models.keys() and isinstance(models[key], tuple):
            models[key] = (model, *models[key])
        else:
            models[key] = model
    
    def get_model_desc(self, models):
        desc = []
        for key, model_info in models.items():
            if isinstance(model_info, tuple):
                desc += [key + ' (' + ' / '.join(model_info[1:]) + ')']
            else:
                desc += [key]
        return '\n'.join(desc)

    def get_deepcopied_models(self, models):
        res = {}
        def try_deepcopy(model):
            try:
                return copy.deepcopy(model)
            except Exception as e:
                print('deepcopy exception: ' + str(e))
                return model
            
        for key, model_info in models.items():
            res[key] = try_deepcopy(model_info) if not isinstance(model_info, tuple) else (try_deepcopy(model_info[0]), *model_info[1:])
        return res

    def to_device(self, models, device):
        res = {}
        def try_to_device(model):
            try:
                return model.to(device)
            except Exception as e:
                print('to device exception: ' + str(e))
                return model
            
        for key, model_info in models.items():
            res[key] = try_to_device(device) if not isinstance(model_info, tuple) else (try_to_device(model_info[0]), *model_info[1:])
        return res


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss1(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False,lb_smooth=0.1):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lb_smooth = lb_smooth
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        pt1 = F.softmax(input, dim=1)
        #print('pt1',pt1)
        logits = input.float()
        num_classes = logits.size(1)
        label = target.clone().detach()

        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
        lb_one_hot = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        #print('pt',pt)
        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            #logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt

        #loss = -torch.sum(lb_one_hot * (1 - pt1) ** self.gamma * torch.log(pt1))
        #loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)* lb_one_hot-(1-self.alpha)*pt**self.gamma*(1-lb_one_hot)*torch.log(pt)
        #loss=-torch.sum(lb_one_hot*(1-pt1)**self.gamma*torch.log(pt1))
        print(loss)


        if self.size_average: return loss.mean()
        else:
         return loss.sum()


def focal_loss11(input_values, gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss1111(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(FocalLoss1111, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
    def forward(self, input, target):
        return focal_loss11(F.cross_entropy(input, target,label_smoothing=0.1), self.gamma)


    #def forward(self, input, target):
    #    return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.gamma)


class FocalLoss111(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss111, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32).cuda()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()

class FocalLoss11(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None,lb_smooth=0.1):
        super(FocalLoss11, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).cuda()
        else:
            self.alpha = alpha
        self.epsilon = epsilon
        self.lb_smooth=lb_smooth

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        logits = input.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()

        lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
        #print('label.unsqueeze(1)', label.unsqueeze(1), ' lb_pos', lb_pos)
        lb_one_hot = torch.empty_like(logits).fill_(
            lb_neg).scatter_(1, label.unsqueeze(1).to(torch.long), lb_pos).detach()
        print
        num_labels = input.size(-1)
        idx = target.view(-1, 1).long()
        one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32).cuda()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        logits = torch.softmax(input, dim=-1)
        #print('one_hot_key :',one_hot_key ,'lb_one_hot',lb_one_hot)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss1 = -self.alpha * lb_one_hot * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        #print('loss:',loss,'loss1',loss1)


        loss = loss.sum(1)
        loss1=loss1.sum(1)
        #print('loss.mean:', loss.mean(), 'loss1.mean', loss1.mean())


        return loss1.mean()

class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    Adapted from https://github.com/CoinCheung/pytorch-loss
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-1):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = input.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = target.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            #print('n_valid:',n_valid,'ignore:',ignore,'label:',label,'label[ignore]:',label[ignore],num_classes)

            label[ignore] = 0
            #print('label[ignore]:',label[ignore])

            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
            #print('lb_one_hot:',lb_one_hot,'logits:',logits)


        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        '''
        logs= torch.empty_like(logits).detach()
        fl=FocalLoss1(gamma=2,
                   alpha=[1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
                          1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18])
        for i in range(0,18):
            ii=torch.empty_like(target).detach()
            ii[0]=i
            #print('ii:',ii)
            logs[0][i]=fl(input,ii.type(torch.long))
        loss=torch.sum(logs * lb_one_hot, dim=1)
        print('logs:',logs,'loss:',loss)
        '''
        #print(loss.sum()/ n_valid)

        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        '''
        j=0
        for x in target[0]:
            print(target[0][j])
        
            if x==1:
                target[0][j]=0.9
                j+=1
            else:
                target[0][j] = 0.1
                j += 1
        '''
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1-target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss



import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import json


class BalancedSoftmax(_Loss):
    """
    Balanced Softmax Loss
    """
    def __init__(self, freq_path):
        super(BalancedSoftmax, self).__init__()
        with open(freq_path, 'r') as fd:
            freq = json.load(fd)
        freq = torch.tensor(freq)
        self.sample_per_class = freq

    def forward(self, input, label, reduction='mean'):
        return balanced_softmax_loss(label, input, self.sample_per_class, reduction)


def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def create_loss(freq_path):
    print('Loading Balanced Softmax Loss.')
    return BalancedSoftmax(freq_path)

from benchmark.data.datasets.registery import longlabel
longlabel=torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0., 0.,1., 1., 1., 0.,0., 0., 0., 0., 0.]], device='cuda:0')
'''
print('longlabel',longlabel)
longlabel=torch.Tensor(longlabel)
longlabel=longlabel.repeat(1, 1)
'''
def iflong(iflonglab,label_source):
    j = 0
    for i in label_source:
        i = int(i)
        # print('i',i==0,j)
        #if i == 0 or i == 1 or i == 9 or i == 10 or i == 11:

        if int(longlabel[0,i])==1:
            iflonglab[j, 0] = 0.
            # print('xx[j,0]',xx[j,0])
        else:
            iflonglab[j, 0] = 1.
        j += 1
    return iflonglab