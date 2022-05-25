from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch


class IncreamentalDualClusterContrastTrainer(object):
    def __init__(self, encoder, loss=None, loss_w=None, loss_=None, model_old=None, fisher=None) -> None:
        super(IncreamentalDualClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.loss = loss
        self.loss_w = loss_w
        self.loss_ = loss_
        self.model_old = model_old
        self.fisher = fisher
        self.regtype = None

    def train(self, epoch, optimizer, data_loader, data_loader_=None, iters=100, print_freq=20, stage=0, regtype=None):
        losses = AverageMeter()
        if regtype != None:
            self.regtype = regtype

        self.encoder.train()   
        for ii in range(iters): 
            if stage==0:
                inputs = data_loader.next()
                inputs, labels, _ = self._parse_data(inputs)
                f_out = self._forward(inputs)
                loss, icc_cluster_features, ccc_cluster_features = self.loss(f_out, labels)

            elif stage==1:
                inputs = data_loader.next()
                inputs_  = data_loader_.next()
                inputs, labels, _ = self._parse_data(inputs)
                inputs_, labels_, _ = self._parse_data(inputs_)
                f_out = self._forward(inputs)
                f_out_ = self._forward(inputs_)
                loss_curr, icc_cluster_features, ccc_cluster_features = self.loss(f_out, labels)
                loss_past, _, _ = self.loss_(f_out_,labels_)
                loss = loss_curr + self.loss_w * loss_past
            
            if self.regtype=='ewc' or self.regtype=='EWC':
                loss = loss + self.ewc_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print log
            if (ii+1)%print_freq==0:
                print('{}->{}:{}'.format(epoch, ii, losses.avg))

    def ewc_loss(self):
        loss_reg = 0.0
        for (name,param),(_,param_old) in zip(self.encoder.named_parameters(),self.model_old.named_parameters()):
            loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
        loss_reg*=5000
        return loss_reg

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _forward(self, inputs):
        return self.encoder(inputs)


class DualClusterContrastTrainer(object):
    def __init__(self, encoder, loss=None,loss1=None,model_old=None,fisher=None):
        super(DualClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.loss = loss
        self.loss1 = loss1
        self.model_old = model_old
        self.fisher = fisher

    def train(self, epoch, data_loader, optimizer, print_freq=10, stage=0):
        self.encoder.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, labels, indexes = self._parse_data(inputs)

            f_out = self._forward(inputs)
            if stage==0:
                loss, icc_centers, ccc_centers= self.loss(f_out, labels)
            else:
                loss, icc_centers, ccc_centers = self.loss(f_out, labels)
                loss_reg = 0.0
                for (name,param),(_,param_old) in zip(self.encoder.named_parameters(),self.model_old.named_parameters()):
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
                loss_reg*=5000
                loss += loss_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
        return icc_centers, ccc_centers

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)




class DualClusterContrastTrainer_(object):
    def __init__(self, model_old=None,fisher=None):
        super(DualClusterContrastTrainer_, self).__init__()
        self.model_old = model_old
        self.fisher = fisher

    def train(self, epoch, data_loader, optimizer, model, model_loss, print_freq=10,stage=0,weight=1.0):
        model.train()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)
            inputs, labels, indexes = self._parse_data(inputs)

            f_out = self._forward(model, inputs)
            if stage==0:
                loss, icc_centers, ccc_centers= model_loss(f_out, labels)
            else:
                loss, icc_centers, ccc_centers = model_loss(f_out, labels)
                loss_reg = 0.0
                for (name,param),(_,param_old) in zip(model.named_parameters(),self.model_old.named_parameters()):
                    loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2
                loss_reg*=10000
                loss += loss_reg

            loss = loss*weight
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
        return icc_centers, ccc_centers

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, model, inputs):
        return model(inputs)

