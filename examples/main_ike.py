# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
from distutils.log import debug
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dualclustercontrast import datasets
from dualclustercontrast import models
from dualclustercontrast.trainers import DualClusterContrastTrainer_, IncreamentalDualClusterContrastTrainer
from dualclustercontrast.evaluators import Evaluator, extract_features, evaluate_loss, load_random_state, save_random_state
from dualclustercontrast.utils.data import IterLoader
from dualclustercontrast.utils.data import transforms as T
from dualclustercontrast.utils.data.sampler import RandomMultipleGallerySampler
from dualclustercontrast.utils.data.preprocessor import Preprocessor
from dualclustercontrast.utils.logging import Logger
from dualclustercontrast.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from dualclustercontrast.utils.faiss_rerank import compute_jaccard_distance
from dualclustercontrast.dataloader import *
from dualclustercontrast.function import *

from debug import *

start_epoch = best_mAP = 0


def modeldict_weighted_average(ws, weights=[]):
    w_avg = {}
    for layer in ws[0].keys():
        w_avg[layer] = torch.zeros_like(ws[0][layer])
    if weights == []: weights = [1.0/len(ws) for i in range(len(ws))]
    for wid in range(len(ws)):
        for layer in w_avg.keys():
            w_avg[layer] = w_avg[layer] + ws[wid][layer] * weights[wid]
    return w_avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0, pooling_type=args.pooling_type)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        
    main_worker(args)


from torch.nn import Parameter
def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip):]
        if name not in tgt_state:
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            print('mismatch:', name, param.size(), tgt_state[name].size())
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    if len(missing) > 0:
        print("missing keys in state_dict:", missing)

    return model


import torch
import torch.nn.functional as F
from torch import nn, autograd
import random
import copy


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

def VCLc(a,b):
    n = a.shape[0]
    mask=list(range(n))
    L2_loss=((a[mask] - b[mask]) ** 2).sum() / ((n) * 2) ## L2_Loss of seen classes
    tot_loss=L2_loss
    return tot_loss

def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic() # set the starting time

    cudnn.benchmark = False

    if args.testbatchsize==True:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log_'+str(args.batch_size)+'.txt'))
    else:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
        
    if args.debug == True:
        print('\n----Runing in Debug Mode----\n')      
    print("==========\nArgs:{}\n==========\n".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    '''
    Dataset statistics:
        ----------------------------------------
        subset   | # ids | # images | # cameras
        ----------------------------------------
        train    |   751 |    12936 |         6
        query    |   750 |     3368 |         6
        gallery  |   751 |    15913 |         6
        ----------------------------------------
    '''

    # Create model
    model = create_model(args)
    ################ Debug Mode ################
    # View the network architecture
    if args.debug == True:
        ViewNetwork(model, args, verbose=False)
    ################ Debug Mode ################
    
    model.cuda()
    # model = nn.DataParallel(model)
    ################ Debug Mode ################
    # View the network architecture
    if args.debug == True:
        sanitycheck(model)
    ################ Debug Mode ################

    # Evaluator
    evaluator = Evaluator(model)
    
    ################ Debug Mode ################
    # View the dataset and data loader 
    if args.debug == True:
        ViewDataset(dataset, model, args, verbose=True)
    ################ Debug Mode ################


    old_models = collections.defaultdict(list)
    freeze_old_models = collections.defaultdict(list)
    fishers = collections.defaultdict(list)


    # Trainer
    cam_centers_icc = collections.defaultdict(list)
    cam_centers_ccc = collections.defaultdict(list)
    cam_labels = collections.defaultdict(list)
    cam_num = collections.defaultdict(list)
    cam_datasets = collections.defaultdict(list)
    cam_prec = collections.defaultdict(list)

    trainer = DualClusterContrastTrainer_()
    if args.debug == True:
        exit(0)

    # view initial mAP
    sys.stdout.flush()
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

    #if ci==0:
    #    cam_orders=[0,1,2,3,4,5]
    #elif ci==1:
    #    cam_orders=[0,3,5,2,1,4] 
    #elif ci==2:
    #    cam_orders=[0,5,4,1,3,2]
    #elif ci==3:
    #    cam_orders=[5,2,3,4,0,1]
    #elif ci==4:
    #    cam_orders=[4,2,0,1,3,5]

    for c in range(6):
        #c = cam_orders[c_]
        best_mAP=0.0
        curr_lr = args.lr
        params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=curr_lr, weight_decay=args.weight_decay)
        if c==0:
            milestones=[15,30,45]
            max_epochs=30
            step_epoch=30
        else:
            milestones=[15,30,45]
            max_epochs=30
            step_epoch=5
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
       
        for epoch in range(max_epochs): 
            lr = lr_scheduler.get_last_lr()[0]
            print('Start camera {} with learning rate:{}'.format(c,lr))
            sys.stdout.flush()
            weight_loss = 1.0
            if epoch%step_epoch==0:
                #sys.stdout.flush()
                init_model = copy.deepcopy(model)
                with torch.no_grad():
                    print('==> Create pseudo labels for unlabeled data')
                    cluster_loader = get_test_loader(dataset, args.height, args.width,args.batch_size, args.workers, testset=sorted(dataset.train_cams[c][0])) 
                    features, labels = extract_features(model, cluster_loader, print_freq=50)
                    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)
                    labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)

                @torch.no_grad()
                def generate_cluster_features(labels, features):
                    centers = collections.defaultdict(list)
                    for i, label in enumerate(labels):
                        if label == -1:
                            continue
                        centers[labels[i]].append(features[i])
                    centers = [
                        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                    ]
                    centers = torch.stack(centers, dim=0)
                    return centers

                pseudo_labels = labels.data.cpu().numpy()
                curr_center_feat = generate_cluster_features(pseudo_labels, features)
                del cluster_loader, features

                if epoch==0:
                    num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                    dcc_loss = DCCLoss(2048,num_cluster,weight=args.w, momentum = args.momentum, init_feat=F.normalize(curr_center_feat, dim=1).cuda())
                    cam_num[c] = num_cluster                

                if c>0:
                    # curr_center_feat, past_center_feat
                    cluster_features = curr_center_feat
                    cluster_features_ = past_center_feat
                    dist = cluster_features.mm(cluster_features_.t())
                    dist = dist.data.cpu().numpy()
                    dist_ = cluster_features_.mm(cluster_features.t())
                    dist_ = dist_.data.cpu().numpy()
                    pseudo_labels_c = copy.deepcopy(pseudo_labels)
                    new_label=-1
                    for i in range(curr_center_feat.shape[0]):
                        i_ = np.argmax(dist[i])
                        min_i = np.argmax(dist_[i_])
                        idx = np.where(pseudo_labels==i)
                        if min_i==i:
                            new_label = new_label+1
                            pseudo_labels_c[idx]=new_label
                            if new_label==0:
                                new_center = past_center_feat[i_].view(1,-1)
                            else:
                                new_center = torch.cat((new_center,past_center_feat[i_].view(1,-1)))
                        else:
                            pseudo_labels_c[idx]=-1
                    print('new_center:{}'.format(new_center.shape))
                    pseudo_labeled_dataset = []
                    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train_cams[c][0]), pseudo_labels_c)):
                        if label != -1:
                            pseudo_labeled_dataset.append((fname, label.item(), cid))
                    nc = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                    nc_ = len(set(pseudo_labels_c)) - (1 if -1 in pseudo_labels_c else 0)
                    past_datasets=pseudo_labeled_dataset
                    
                    loss_w = nc_/nc
                    print(loss_w)
                    print('==> For camera {}, Statistics for epoch {}: {} / {} clusters'.format(c,epoch,nc_, curr_center_feat.shape[0]))

            model.train()
            if c==0:
                train_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, iters, trainset = sorted(dataset.train_cams[c][0]))               
                losses = AverageMeter()
                for ii in range(iters):
                    inputs = train_loader.next()
                    inputs, labels, _ = _parse_data(inputs)
                    f_out = model(inputs)[0]
                    loss = dcc_loss(f_out,labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses.update(loss.item())
                    del f_out
                    if (ii+1)%args.print_freq==0:
                        print('{}->{}:{}'.format(epoch, ii, losses.avg))

            else:
                num_cluster_ = new_center.shape[0]
                dcc_loss_ = DCCLoss2(2048,num_cluster_,weight= args.w, init_icc_feat=F.normalize(new_center, dim=1).cuda(),init_ccc_feat=F.normalize(new_center, dim=1).cuda())
                train_loader_ = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, iters, trainset=past_datasets)
                train_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, iters, trainset = sorted(dataset.train_cams[c][0]))
                losses_curr = AverageMeter()
                losses_past = AverageMeter()
                losses_kd = AverageMeter()
                for ii in range(iters):
                    inputs = train_loader.next()
                    inputs_  = train_loader_.next()
                    inputs, labels, _ = _parse_data(inputs)
                    inputs_, labels_, _ = _parse_data(inputs_)
                    f_out_curr, f_mid_1_curr, f_mid_2_curr,f_mid_3_curr = model(inputs)
                    f_out_past, f_mid_1_past, f_mid_2_past, f_mid_3_past = model(inputs_) 
                    
                    f_out_past_, f_mid_1_past_, f_mid_2_past_, f_mid_3_past_ = freeze_model_old(inputs_)
                    loss_kd_0 = VCLc(f_out_past,f_out_past_.detach())
                    loss_kd_2 = VCLc(f_mid_2_past,f_mid_2_past_.detach())
                    loss_kd_3 = VCLc(f_mid_3_past,f_mid_3_past_.detach())

                    loss_kd = loss_kd_0 + 0.25*loss_kd_2+0.25*loss_kd_3

                    loss_curr  = dcc_loss(f_out_curr,labels)
                    loss_past = dcc_loss_(f_out_past,labels_)

                    loss = loss_kd + loss_w*loss_curr + loss_past

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses_curr.update(loss_curr.item())
                    losses_past.update(loss_past.item())
                    losses_kd.update(loss_kd.item())
                    if (ii+1)%args.print_freq==0:
                        print('{}->{}:curr:{},past:{},kd:{}'.format(epoch, ii, losses_curr.avg,losses_past.avg,losses_kd.avg))

            if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
                mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
                is_best = (mAP > best_mAP)
                best_mAP = max(mAP, best_mAP)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
                
                cam_prec[c] = best_mAP

                print('\n * Finished epoch {:3d} Moment:{}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                    format(epoch, args.momentum,  mAP, best_mAP, ' *' if is_best else ''))

            if epoch == max_epochs-1:
                ################################################################################################################################################
                #checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
                #model.load_state_dict(checkpoint['state_dict'])
                with torch.no_grad():
                    print('==> Create pseudo labels for unlabeled data')
                    cluster_loader = get_test_loader(dataset, args.height, args.width,args.batch_size, args.workers, testset=sorted(dataset.train_cams[c][0]))
                    features, labels = extract_features(model, cluster_loader, print_freq=50)
                    features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)
                    labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)

                @torch.no_grad()
                def generate_cluster_features(labels, features):
                    centers = collections.defaultdict(list)
                    for i, label in enumerate(labels):
                        if label == -1:
                            continue
                        centers[labels[i]].append(features[i])
                    centers = [
                        torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                    ]
                    centers = torch.stack(centers, dim=0)
                    return centers

                pseudo_labels = labels.data.cpu().numpy()
                curr_center_feat = generate_cluster_features(pseudo_labels, features)
                curr_center_feat = F.normalize(curr_center_feat, dim=1)
                ###################################################################################################################################################
                if 1:
                    if c==0:
                        past_center_feat = curr_center_feat
                    else: # combine paster_center_feat and curr_center_feat
                        p_num = past_center_feat.shape[0]
                        c_num = curr_center_feat.shape[0]
                        p2c_dist = past_center_feat.mm(curr_center_feat.t())
                        p2c_dist = p2c_dist.data.cpu().numpy()
                        c2p_dist = curr_center_feat.mm(past_center_feat.t())
                        c2p_dist = c2p_dist.data.cpu().numpy()

                        for i in range(c_num):
                            i_ = np.argmax(c2p_dist[i])
                            min_i = np.argmax(p2c_dist[i_])
                            if min_i == i:
                                past_center_feat[i_] = args.weight*past_center_feat[i_]+(1-args.weight)*curr_center_feat[i]
                                past_center_feat[i_] /= past_center_feat[i_].norm()
                            else:
                                past_center_feat = torch.cat((past_center_feat,curr_center_feat[i].view(1,-1)))

                print('past_center_shape:{}'.format(past_center_feat.shape))
                ###################################################################################################################################################

                freeze_model_old = copy.deepcopy(model)
                freeze_model_old.eval()
                freeze_model(freeze_model_old)

            lr_scheduler.step()
        print(cam_prec)

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def _parse_data(inputs):
    imgs, _, pids, _, indexes = inputs
    return imgs.cuda(), pids.cuda(), indexes.cuda()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dual Cluster Contrastive Learning for person re-id")
    # debug
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--testbatchsize', type=bool, default=False)
    parser.add_argument('--regular-weight', type=float, default=5000)
    parser.add_argument('--icm', type=bool, default=False)
    parser.add_argument('--end-cm', type=bool, default=False)
    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--weight', type=float, default=0.0)


    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.0,
                        help="update momentum for the hybrid memory")
    
    parser.add_argument('--w', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--step-size', type=int, default=50)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                            default=osp.join(working_dir, 'examples/data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs_0914_adapt'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    main()
