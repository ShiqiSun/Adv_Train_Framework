device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0
model = dict(
    type = "WideResNet",
    depth = 28,
    num_classes = 1000,
    widen_factor = 10,
    dropRate = None,
    stride_list = [2, 2, 2, 2]
)

dataset = dict(
    type = "ImageNet",   
    root_dir = None,
    train = dict(
        batch_size = 256,
    ),
    test = dict(
        batch_size = 256,
    ),
    normalize = dict(
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 1500,
    save_freq = 10,
    epoch = 300,
    lr_adjust_list = [50, 70, 90, 110],
    optim = dict(
        type="SGD",
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 2e-4
    ),
    loss =  dict(
        type = "CrossEntropy"
    ),
    mode = {
        "train": 5,
        "eval": 1
    }
)

log_config = dict(
    log_level = "info",
    log_dir = None,
    when = 'D',
    backCount = 3
)