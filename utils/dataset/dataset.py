import os

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from utils.register.registers import DATASETS

default_dataset_path = '/data_volumn_2'

@DATASETS.register
def CIFAR10(cfg):
    train_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path
    test_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path

    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    
    mean = (cfg.dataset.normalize.mean !=  None) and cfg.dataset.normalize.mean \
                        or [0.485, 0.456, 0.406]
    std = (cfg.dataset.normalize.std !=  None) and cfg.dataset.normalize.std \
                        or [0.229, 0.224, 0.225]


    
    if cfg.dataset.normalize.is_normalized:
        normalize = transforms.Normalize(mean=mean,
                                     std=std)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.ToTensor(),
        ])

    train_set = datasets.CIFAR10(root=train_dataset_path, train=True, download=True, transform=transform_train)
    
    if cfg.is_distributed:
        train_sampler = DistributedSampler(train_set)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler,batch_size=cfg.dataset.train.batch_size, shuffle=shuffle_flag, **kwargs)
    testset = datasets.CIFAR10(root=test_dataset_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.dataset.test.batch_size, shuffle=False, **kwargs)
    dataloader = {
        "train":train_loader,
        "test":test_loader,
        "val":None
    }

    return dataloader

@DATASETS.register
def ImageNet(cfg):
    train_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path + '/ILSVRC2012'
    test_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path + '/ILSVRC2012'
    mean = (cfg.dataset.normalize.mean !=  None) and cfg.dataset.normalize.mean \
                        or [0.485, 0.456, 0.406]
    std = (cfg.dataset.normalize.std !=  None) and cfg.dataset.normalize.std \
                        or [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    traindir = os.path.join(train_dataset_path, 'train')
    testdir = os.path.join(test_dataset_path, 'val')


    train_set = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    if cfg.is_distributed:
        train_sampler = DistributedSampler(train_set)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=train_sampler, batch_size=cfg.dataset.train.batch_size, shuffle=shuffle_flag,
        num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=cfg.dataset.test.batch_size, shuffle=False,
        num_workers=1)

    dataloader = {
        "train":train_loader,
        "test":test_loader,
        "val":None
    }

    return dataloader

@DATASETS.register
def MNIST(cfg):
    train_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path
    test_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path
    train_set = datasets.MNIST(train_dataset_path, train=True, download=True,
                       transform=transforms.ToTensor())
    if cfg.is_distributed:
        train_sampler = DistributedSampler(train_set)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True
    train_loader = torch.utils.data.DataLoader( 
        train_set,
        sampler = train_sampler,
        batch_size=cfg.dataset.train.batch_size, shuffle=shuffle_flag)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(test_dataset_path, train=False,
                       transform=transforms.ToTensor()),
        batch_size=cfg.dataset.test.batch_size, shuffle=False)

    dataloader = {
        "train":train_loader,
        "test":test_loader,
        "val":None
    }

    return dataloader

@DATASETS.register
def CIFAR10_resize(cfg):
    train_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path
    test_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path

    kwargs = {'num_workers': 1, 'pin_memory': True} if True else {}
    
    mean = (cfg.dataset.normalize.mean !=  None) and cfg.dataset.normalize.mean \
                        or [0.485, 0.456, 0.406]
    std = (cfg.dataset.normalize.std !=  None) and cfg.dataset.normalize.std \
                        or [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean,
                                     std=std)

    Image_size = cfg.dataset.image_size
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((Image_size, Image_size), scale=(0.05, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop((Image_size, Image_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        normalize
    ])

    train_set = datasets.CIFAR10(root=train_dataset_path, train=True, download=True, transform=transform_train)
    
    if cfg.is_distributed:
        train_sampler = DistributedSampler(train_set)
        shuffle_flag = False
    else:
        train_sampler = None
        shuffle_flag = True

    train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sampler,batch_size=cfg.dataset.train.batch_size, shuffle=shuffle_flag, **kwargs)
    testset = datasets.CIFAR10(root=test_dataset_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=cfg.dataset.test.batch_size, shuffle=False, **kwargs)
    dataloader = {
        "train":train_loader,
        "test":test_loader,
        "val":None
    }

    return dataloader