We Provide a Machine Learning Training Framework, Which you could customize your own dataset, optimizers, models, loss and training mode. Here is some guideline.

***

# tools: 

##  1. Train:
###  python -m torch.distributed.launch --nproc_per_node=2 ./tools/trainval/train.py --config CONFIG_PATH
###  run _tools/train.py_ to train a classifier

### train:
    log_interval = 40,
    save_freq = 10,
    epoch = 300,
    lr_adjust_list = [30, 50, 70, 90],
    resume = dict(
        is_resume = True,
        resume_from_work_dir = True,
        resume_from_file = None 
    ),
    optim : see optimizer part
    loss : see loss part
    mode = {"train": 10, "eval": 1}


***

# Models

## 1. WideResnet:
    type = "WideResNet"
    depth = 28 customize para
    num_classes=10 output classes
    widen_factor = 10
    dropRate = None 
    stride_list = [1, 1, 2, 2] for CIFAR10 [2, 2, 2, 2] for ImageNet

***
# Loss:

## 1. CrossEntropy:

    type = "CrossEntropy"


***
# Optimizer:

## 1. SGD:
    type="SGD",
    lr = 0.1,
    momentum = 0.9,
    weight_decay = 2e-4

***
# Dataset:

## 1. MNIST, CIFAR10, ImageNet
    type = "MNIST", "CIFAR10" or  "ImageNet",
    root_dir = None,
    train = dict(
        batch_size = 512,
    ),
    test = dict(
        batch_size = 512,
    ),
    normalize = dict(
        mean = None,
        std = None
    )