from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
from datetime import timedelta

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch
from .dataloader import *
from .function import *

def extract_cnn_feature(model, inputs):
    inputs1 = to_torch(inputs).cuda()
    outputs1 = model(inputs1)[0]
    outputs1 = outputs1.data.cpu()

    inputs2 = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
    inputs2 = to_torch(inputs2).cuda()
    outputs2 = model(inputs2)[0]
    outputs2 = outputs2.data.cpu()

    ff = outputs1+outputs2
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    outputs = ff.div(fnorm.expand_as(ff))

    return outputs


def extract_features(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m=torch.addmm(input=dist_m, mat1=x, mat2=y.t(), beta=1,alpha=-2)
    dist_m.addmm_(mat1=x, mat2=y.t(), beta=1, alpha=-2)

    return dist_m, x.numpy(), y.numpy()
    


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True),}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(features, query, gallery)
        results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


def _parse_data(inputs):
    imgs, _, pids, _, indexes = inputs
    return imgs.cuda(), pids.cuda(), indexes.cuda()


def save_random_state():
    # save the random state
    state1 = random.getstate()
    state2 = np.random.get_state()      
    state3 = torch.random.get_rng_state() 
    return state1, state2, state3


def load_random_state(state1, state2, state3):
    # set the random state back
    random.setstate(state1)
    np.random.set_state(state2)
    torch.random.set_rng_state(state3)


def evaluate_loss(args, cam_num, dataset, init_model, model, optimizer, loss_func=None):
    # save the random state
    state1 = random.getstate()
    state2 = np.random.get_state()      
    state3 = torch.random.get_rng_state() 

    print('\nEvaluate loss ')
    start_time = time.monotonic()
    for cam_id in range(cam_num):
        with torch.no_grad():
            cluster_loader = get_test_loader(dataset, args.height, args.width,args.batch_size, args.workers, testset=sorted(dataset.train_cams[cam_id][0])) 
            features, labels = extract_features(init_model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[cam_id][0])], 0)
            labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[cam_id][0])], 0)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            # print('the dimension of centers is {}'.format(len(centers)))
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)
            return centers
            
        pseudo_labels = labels.data.cpu().numpy()
        curr_center_feat = generate_cluster_features(pseudo_labels, features)
        del cluster_loader, features

        num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        loss_func = DCCLoss(2048, num_cluster, weight=args.w, momentum = args.momentum, init_feat=F.normalize(curr_center_feat, dim=1).cuda())

        model.eval()
        eval_losses = AverageMeter()
        val_loader = get_val_loader(dataset, args.height, args.width, batch_size=128, workers=args.workers, trainset = sorted(dataset.train_cams[cam_id][0]), type=1)
        for i_batch, batch_data in enumerate(val_loader):
            inputs, labels, _ = _parse_data(batch_data)
            f_out = model(inputs)
            loss, icc_cluster_features, ccc_cluster_features = loss_func(f_out,labels)
            optimizer.zero_grad()
            loss.backward()
            eval_losses.update(loss.item())
        print('Evaluate loss in camera {}:{} with mode 1'.format(cam_id, eval_losses.avg))
    
        model.eval()
        eval_losses = AverageMeter()
        val_loader = get_val_loader(dataset, args.height, args.width, batch_size=128, workers=args.workers, trainset = sorted(dataset.train_cams[cam_id][0]), type=2)
        for i_batch, batch_data in enumerate(val_loader):
            inputs, labels, _ = _parse_data(batch_data)
            f_out = model(inputs)
            loss, icc_cluster_features, ccc_cluster_features = loss_func(f_out,labels)
            optimizer.zero_grad()
            loss.backward()
            eval_losses.update(loss.item())
        print('Evaluate loss in camera {}:{} with mode 2'.format(cam_id, eval_losses.avg))
        del val_loader

        model.eval()
        eval_losses = AverageMeter()
        val_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, args.iters, trainset = sorted(dataset.train_cams[cam_id][0]))
        for ii in range(args.iters):
            inputs = val_loader.next()
            inputs, labels, _ = _parse_data(inputs)
            f_out = model(inputs)
            loss,icc_cluster_features, ccc_cluster_features = loss_func(f_out,labels)
            optimizer.zero_grad()
            loss.backward()
            eval_losses.update(loss.item())
        print('Evaluate loss in camera {}:{} with mode 4'.format(cam_id, eval_losses.avg))
        del val_loader

    # set the random state back
    random.setstate(state1)
    np.random.set_state(state2)
    torch.random.set_rng_state(state3)

    end_time = time.monotonic()
    print('Evaluate loss running time:', timedelta(seconds=end_time - start_time), '\n')
    
