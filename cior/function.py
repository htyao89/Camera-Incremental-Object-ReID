import torch
import torch.nn.functional as F
from torch import nn, autograd
import random
import collections

class DCC(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut,  momentum):
        ctx.lut = lut
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.lut.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs,targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.lut)

        for x, y in zip(inputs,targets.data.cpu().numpy()):
            ctx.lut[y] = ctx.lut[y] * ctx.momentum + (1 - ctx.momentum) * x
            ctx.lut[y] /= ctx.lut[y].norm()

        return grad_inputs, None, None, None


def oim(inputs, targets, lut, momentum=0.1):
    return DCC.apply(inputs, targets, lut, torch.Tensor([momentum]).to(inputs.device))

import copy
class CurrLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True, reduction='mean', init_feat=[],c=0):
        super(CurrLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

        self.register_buffer('lut', torch.zeros(num_classes, num_features).cuda())
        self.lut = copy.deepcopy(init_feat)

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        return loss


class EWCLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=0.0,
                 weight=None, size_average=True, reduction='mean', init_icc_feat=[], init_ccc_feat=[]):
        super(EWCLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction
        
        self.register_buffer('lut_ewc', torch.zeros(num_classes, num_features).cuda())
        self.lut_ewc = copy.deepcopy(init_icc_feat)

    def forward(self, inputs, targets):
        inputs = oim(inputs, targets, self.lut_ewc, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        return loss

class PastLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=20.0, momentum=1.0,
                 weight=None, size_average=True, reduction='mean', init_icc_feat=[], init_ccc_feat=[]):
        super(PastLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average
        self.reduction = reduction

        self.register_buffer('lut_past', torch.zeros(num_classes, num_features).cuda())
        self.lut_past = copy.deepcopy(init_icc_feat)

    def forward(self, inputs, targets):
        inputs= oim(inputs, targets, self.lut_past, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, reduction=self.reduction)
        return loss
