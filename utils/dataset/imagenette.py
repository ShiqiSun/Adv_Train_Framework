from cgi import test
import os

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

from utils.register.registers import DATASETS

default_dataset_path = '/data_volumn_2'

@DATASETS.register
def ImageNette(cfg):

    train_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                        or default_dataset_path + '/imagenette/imagenette2'
    test_dataset_path = (cfg.dataset.root_dir != None) and cfg.dataset.root_dir \
                         or default_dataset_path + '/imagenette/imagenette2'
    mean = (cfg.dataset.normalize.mean !=  None) and cfg.dataset.normalize.mean \
                        or [0.485, 0.456, 0.406]
    std = (cfg.dataset.normalize.std !=  None) and cfg.dataset.normalize.std \
                        or [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean,
                                    std=std)
    print(mean, std)

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

    test_set =  datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=cfg.dataset.test.batch_size, shuffle=False,
        num_workers=1)
    
    dataloader = {
        "train":train_loader,
        "test":test_loader,
        "val":None
    }

    return dataloader

# def sample_by_class(dataset, sample_list):
#     sample_data_set = []
#     for i in range(len(dataset)):
#         img, label = dataset[i]
#         if label in sample_list:
#             sample_data_set.append([img, label])
#         if i%1000 == 0:
#             print(f"{i}/{len(dataset)}")
#     sample_dataset = ImageNetSample(sample_data_set)
#     return sample_dataset

# class ImageNetSample(Dataset):
#     def __init__(self, sample_set) -> None:
#         super().__init__()
#         self.sample_set = sample_set
    
#     def __len__(self):
#         return len(self.sample_set)

#     def __getitem__(self, idx):
#         return self.sample_set[idx][0], self.sample_set[idx][1]

    
