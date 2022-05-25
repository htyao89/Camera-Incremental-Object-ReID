from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import os.path as osp
import torch
import random
import copy

from torch.utils.data import DataLoader
from PIL import Image
from . import datasets
from .utils.data import IterLoader
from .utils.data import transforms as T
from .utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySampler1
from .utils.data.preprocessor import Preprocessor


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    print('the dataset root is ' + root)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)


    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation= T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def get_val_loader(dataset, height, width, batch_size, workers, trainset=None, type=1):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    val_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    if type==1:
        num_instances = 16
        train_set = trainset
        rmgs_flag = num_instances > 0
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None
    elif type==2:
        train_set = trainset
        sampler = None
    elif type==3:
        num_instances = 16
        train_set = trainset
        rmgs_flag = num_instances > 0
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler1(train_set, num_instances)
        else:
            sampler = None
    
    val_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=val_transformer),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=False, pin_memory=True, drop_last=True)

    return val_loader


def get_eval_loss_loader(dataset, height, width, batch_size, workers, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    eval_loss_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = trainset
    eval_loss_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=eval_loss_transformer),
            batch_size=batch_size, num_workers=workers, sampler=None,
            shuffle=False, pin_memory=True, drop_last=False)
    
    return eval_loss_loader


def get_eval_loss_loader1(dataset, height, width, batch_size, workers, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    num_instances = 16
    train_set = trainset
    rmgs_flag = num_instances > 0
    sampler = RandomMultipleGallerySampler1(train_set, num_instances)
    
    eval_loss_loader = DataLoader(
            Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
            batch_size=batch_size, num_workers=workers, sampler=sampler,
            shuffle=False, pin_memory=True, drop_last=False)

    return eval_loss_loader
