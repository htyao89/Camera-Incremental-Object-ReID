# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
# import matplotlib.pyplot as plt

import re
import glob
import random
import torch
import collections

import torch.nn.functional as F
from dualclustercontrast import datasets
from dualclustercontrast.dataloader import *
from dualclustercontrast.evaluators import Evaluator, extract_features


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


def ViewNetwork(model, args, verbose=False):
    '''
    View the network architecture

    Parameters
    
    model : 
        the model that has been created
    args : 
        the arguments that has been assigned from the command line
    verbose : bool
        choose wether to print the whole architecture of the network\\
        see He_Deep_Residual_Learning_CVPR_2016_paper for more details(https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)
    '''
    print('\n{} Architecture:'.format(args.arch.capitalize()))
    print('pooling_type: {}'.format(args.pooling_type))
    if verbose==True:
        print(model.base)
    else:
        print('the conv5_x layer of the network')
        print(model.base[-1])


def sanitycheck(model):
    for name, param in model.named_parameters():
        print('name:{0:30}\t param size:'.format(name), param.size())
        

def _parse_data(inputs):
    imgs, _, pids, _, indexes = inputs
    return imgs.cuda(), pids.cuda(), indexes.cuda()


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


def calculate_accum_ids(dataset, cam_num) -> list:
    '''
    for each serial number of cameras, calculate the number of persons being captured by this and 
    previous cameras.

    return
    --------
    a list which each item represents the number of persons being captured by this and previous cameras
    '''
    accum_ids = []
    img_paths = glob.glob(osp.join(dataset.train_dir, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for cam_idx in range(cam_num):   
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            camid -=1
            if camid!=cam_idx:
                continue
            pid_container.add(pid)
        accum_ids.append(len(pid_container))
    
    return accum_ids


def ViewDataset(dataset, model, args, verbose=False):
    '''
    View the architecture of dataset and dataloader

    .. some examples
    Examples
    --------
    An overview of the dataset when `relabel=True`, the second element is `label` obtained by `pid2label[pid]`, where `pid2label` is a `set`
    and each `pid` corresponding to one `label` in an ascending order
    >>> dataset.train_cams[c][0]
    # in forms of (img_path, pid/label, camid)
    [('dir_path/0002_c1s2_000841_01.jpg', 0, 0), ('dir_path/1458_c1s6_009541_02.jpg', 635, 0), ('dir_path/0002_c1s1_000551_01.jpg', 0, 0), ('dir_path/0332_c1s5_051616_05.jpg', 134, 0)]
    
    An overview of the dataset when `relabel=False`, the second element is `pid`
    >>> dataset.train_cams[c][0]
    # in forms of (img_path, pid/label, camid)
    [('dir_path/0002_c1s2_000841_01.jpg', 2, 0), ('dir_path/1458_c1s6_009541_02.jpg', 1458, 0), ('dir_path/0002_c1s1_000551_01.jpg', 2, 0), ('dir_path/0332_c1s5_051616_05.jpg', 332, 0)]
    
    An overview of the sorted dataset, `relabel` is defaulted to be `True`
    >>> sorted(dataset.train_cams[0][0])
    # in forms of (img_path, pid/label, camid)
    [('dir_path/0002_c1s1_000451_03.jpg', 0, 0), ('dir_path/0002_c1s1_000551_01.jpg', 0, 0), ('dir_path/0007_c1s6_028546_01.jpg', 1, 0), ('dir_path/0007_c1s6_028546_04.jpg', 1, 0)]
    '''
    cam_num = 6
    print('\n==> Viewing dataset and data loader')
    if verbose==True:
        print('\nAn example overview of the dataset architecture when relabel=True:')
        for i in range(5):
            print(dataset.train_cams[0][0][i])  
        print('\nAn example overview of the sorted dataset architecture:')
        for i in range(10):
            print(sorted(dataset.train_cams[0][0])[i])

    ################ Detailed statistical information of the dataset ################
    if verbose==True:
        print('\n=> Detailed statistical information of the dataset')
        total_num = 0
        for c in range(cam_num):
            print('the number of images taken by camera {} is {}'.format(c, len(dataset.train_cams[c][0])))
            total_num += len(dataset.train_cams[c][0])
        print('the total number of training images in folder bounding_box_train is {}'.format(total_num))

        ################ Information of the data loader ################
        print('\n=> Detailed statistical information of the data loader')
        test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
        # the length of the data loader should be args.batchsize times less than the length of the corresponding dataset
        print('the length of the test loader is {}'.format(len(test_loader)))
        for c in range(cam_num):
            cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, testset=sorted(dataset.train_cams[c][0])) 
            print('the length of the camera {} cluster loader is {}'.format(c, len(cluster_loader)))
        print('\n=> Detailed statistical information of the train loader')
        for c in range(6):
            train_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, args.iters, trainset = sorted(dataset.train_cams[c][0]))
            print('the length of the camera {} train loader is {}'.format(c, len(train_loader)))
        ################ Information of the data loader ################

        cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, testset=sorted(dataset.train_cams[0][0])) 
        features, labels = extract_features(model, cluster_loader, print_freq=50)

        print('\n=> An example item of the features and labels (randomly selected)')
        tmp = random.randint(0, 100) # choose a random item to exhibit
        for fname, feature in features.items():
            # fname = 'dir_name/img.jpg', feature = tensor(2048)
            if tmp == 0:
                print(osp.basename(fname))
                print(feature.shape)
                print(labels[fname])
                break
            tmp -= 1
        del test_loader, cluster_loader, train_loader, features, labels
        ################ Detailed statistical information of the dataset ################

    ids = [] # the number of persons being captured by each camera
    accum_ids = [] # the number of persons being captured until now
    images = [] # the number of images
    num_cluster = []

    for c in range(cam_num):
        cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers, testset=sorted(dataset.train_cams[c][0])) 
        features, labels = extract_features(model, cluster_loader, print_freq=50)
        tmp = 0
        for fname, label in labels.items():
            # calculate the number of persons being captured by each camera
            if tmp < label:
                tmp = label
        features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)
        labels = torch.cat([labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train_cams[c][0])], 0)

        ids.append(tmp.item() + 1)
        images.append(len(dataset.train_cams[c][0]))
        accum_ids = calculate_accum_ids(dataset, cam_num)

        if verbose==True:
            print('\n---camera {}---'.format(c))
            print('the number of persons being observed by camera {} is {}'.format(c, tmp.item() + 1))
            print('the size of features is {}'.format(features.shape))
            print('the size of labels is {}'.format(labels.shape))

        pseudo_labels = labels.data.cpu().numpy()
        curr_center_feat = generate_cluster_features(pseudo_labels, features)
        num_cluster.append(len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0))
        if verbose==True:
            print('the size of pseudo_labels is {}'.format(len(pseudo_labels)))
            print('the size of curr_center_feat is {}'.format(curr_center_feat.shape))
        del cluster_loader, features, labels, pseudo_labels, curr_center_feat

        ################ Observe train loader ################
        train_loader = get_train_loader(args, dataset, args.height, args.width, args.batch_size, args.workers, args.num_instances, args.iters, trainset = sorted(dataset.train_cams[0][0]))
        if verbose==True:
            for ii in range(args.iters):
                inputs = train_loader.next()
                inputs, labels, indexes = _parse_data(inputs)
                f_out = model(inputs)
                if ii==0:
                    print('-Observe train loader:')
                    print('the size of inputs is {}'.format(inputs.shape))
                    print('the size of labels is {}'.format(labels.shape))
                    # print(labels)
                    print('the size of the model output is {}'.format(f_out.shape))
        ################ Observe train loader ################
        del train_loader, inputs, labels, f_out

        ################ Observe evaluate loss loader ################
        batch_size = 32
        loadersize = 0
        eval_loss_loader = get_eval_loss_loader(dataset, args.height, args.width, batch_size=batch_size, workers=args.workers, trainset = sorted(dataset.train_cams[c][0]))
        for i_batch, batch_data in enumerate(eval_loss_loader):
            loadersize = i_batch + 1
            if i_batch==0:
                inputs = batch_data
            # _, labels, indexes = _parse_data(batch_data)
            # if i_batch<=5:
            #     print(labels)
        inputs, labels, indexes = _parse_data(inputs)
        if verbose==True:
            print('-Observe evaluate loss loader:')
            print('the size of evaluate loss loader is {} with batchsize {}'.format(loadersize, batch_size))
            print('the data size of evaluate loss loader is {}'.format(inputs.shape))
        del eval_loss_loader
        ################ Observe evaluate loss loader ################
        
        ################ Observe validation loader ################
        batch_size = 32
        loadersize = 0
        eval_loss_loader = get_eval_loss_loader1(dataset, args.height, args.width, batch_size=batch_size, workers=args.workers, trainset = sorted(dataset.train_cams[c][0]))
        for i_batch, batch_data in enumerate(eval_loss_loader):
            loadersize = i_batch + 1
            if i_batch==0:
                inputs = batch_data
        inputs, labels, indexes = _parse_data(inputs)
        if verbose==True:
            print('-Observe validation loader:')
            print('the size validation loader is {} with batchsize {}'.format(loadersize, batch_size))
            print('the data size of validation loader is {}'.format(inputs.shape))
        del eval_loss_loader
        ################ Observe validation loader ################

    from prettytable import PrettyTable
    print('\nTable statistic:')
    table = PrettyTable(['camid', '# ids', '# accum_ids' ,'# images', '# num_cluster'])
    for c in range(cam_num):
        table.add_row([c, ids[c], accum_ids[c], images[c], num_cluster[c]])
    print(table)
