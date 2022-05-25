import torch
import torch.nn.functional as F
from torch import nn, autograd
import random
import collections

class DCC(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut_ccc, lut_icc,  momentum):
        ctx.lut_ccc = lut_ccc
        ctx.lut_icc = lut_icc
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs_ccc = inputs.mm(ctx.lut_ccc.t())
        outputs_icc = inputs.mm(ctx.lut_icc.t())

        return outputs_ccc,outputs_icc

    @staticmethod
    def backward(ctx, grad_outputs_ccc, grad_outputs_icc):
        inputs,targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs_ccc.mm(ctx.lut_ccc)+grad_outputs_icc.mm(ctx.lut_icc)

        '''
        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.data.cpu().numpy()):
            batch_centers[index].append(instance_feature)

        for y, features in batch_centers.items():
            mean_feature = torch.stack(batch_centers[y],dim=0)
            non_mean_feature = mean_feature.mean(0)
            x = F.normalize(non_mean_feature,dim=0)
            ctx.lut_ccc[y] = ctx.momentum * ctx.lut_ccc[y] + (1.-ctx.momentum) * x
            ctx.lut_ccc[y] /= ctx.lut_ccc[y].norm()

        del batch_centers 
        '''
        for x, y in zip(inputs,targets.data.cpu().numpy()):
            ctx.lut_icc[y] = ctx.lut_icc[y] * ctx.momentum + (1 - ctx.momentum) * x
            ctx.lut_icc[y] /= ctx.lut_icc[y].norm()

        return grad_inputs, None, None, None, None


def oim(inputs, targets, lut_ccc, lut_icc, momentum=0.1):
    return DCC.apply(inputs, targets, lut_ccc, lut_icc, torch.Tensor([momentum]).to(inputs.device))

import copy
class DCCLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True, reduction='mean', init_feat=[],c=0):
        super(DCCLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

        self.register_buffer('lut_ccc_{}'.format(c), torch.zeros(num_classes, num_features).cuda())
        self.lut_ccc = copy.deepcopy(init_feat)

        self.register_buffer('lut_icc_{}'.format(c), torch.zeros(num_classes, num_features).cuda())
        self.lut_icc = copy.deepcopy(init_feat)

    def forward(self, inputs, targets):
        inputs_ccc,inputs_icc = oim(inputs, targets, self.lut_ccc, self.lut_icc, momentum=self.momentum)
        inputs_icc *= self.scalar
        loss = F.cross_entropy(inputs_icc, targets, reduction=self.reduction)
        return loss, self.lut_icc, self.lut_icc


import copy
class DCCJointLoss(nn.Module):
    def __init__(self, momentum=0.0,scalar=20.0):
        super(DCCJointLoss, self).__init__()
        self.momentum = momentum
        self.scalar = scalar

    def forward(self, inputs, targets, lut):
        _,inputs_icc = oim(inputs, targets, lut, lut, momentum=self.momentum)
        inputs_icc *= self.scalar
        loss = F.cross_entropy(inputs_icc, targets, reduction='mean')
        return loss

class DCCLoss1(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True, reduction='mean', init_icc_feat=[], init_ccc_feat=[]):
        super(DCCLoss1, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction
        
        self.register_buffer('lut_ccc1', torch.zeros(num_classes, num_features).cuda())
        self.lut_ccc1 = copy.deepcopy(init_icc_feat)
        self.register_buffer('lut_icc1', torch.zeros(num_classes, num_features).cuda())
        self.lut_icc1 = copy.deepcopy(init_ccc_feat)
        #print('Weight:{},Momentum:{}'.format(self.weight,self.momentum))

    def forward(self, inputs, targets):
        inputs_ccc,inputs_icc = oim(inputs, targets, self.lut_ccc1, self.lut_icc1, momentum=self.momentum)
        inputs_icc *= self.scalar
        _,targets = torch.max(inputs_icc.data,1)  
        # loss = F.cross_entropy(inputs_icc, targets, size_average=self.size_average)
        loss = F.cross_entropy(inputs_icc, targets, reduction=self.reduction)
        return loss, self.lut_icc1, self.lut_icc1

class DCCLoss2(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=1.0,
                 weight=None, size_average=True, reduction='mean', init_icc_feat=[], init_ccc_feat=[]):
        super(DCCLoss2, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

        self.register_buffer('lut_ccc2', torch.zeros(num_classes, num_features).cuda())
        self.lut_ccc2 = copy.deepcopy(init_icc_feat)

        self.register_buffer('lut_icc2', torch.zeros(num_classes, num_features).cuda())
        self.lut_icc2 = copy.deepcopy(init_ccc_feat)

        #print('Weight:{},Momentum:{}'.format(self.weight,self.momentum))

    def forward(self, inputs, targets):
        inputs_ccc,inputs_icc = oim(inputs, targets, self.lut_ccc2, self.lut_icc2, momentum=self.momentum)
        inputs_icc *= self.scalar
        # loss = F.cross_entropy(inputs_icc, targets, size_average=self.size_average)
        loss = F.cross_entropy(inputs_icc, targets, reduction=self.reduction)
        return loss, self.lut_icc2, self.lut_icc2
