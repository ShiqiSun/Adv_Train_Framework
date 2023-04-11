device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0

model = dict(
    type = "SmallCNN",
    num_channels = 3,
    num_classes = 10,
    dropRate = None
)

dataset = dict(
    type = "CIFAR10",   
    root_dir = None,
    train = dict(
        batch_size = 256,
    ),
    test = dict(
        batch_size = 256,
    ),
    normalize = dict(
        is_normalized = False,
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 40,
    save_freq = 100,
    epoch = 300,
    lr_adjust_list = [30, 50, 70, 90],
    resume = dict(
        is_resume = False,
        resume_from_work_dir = False,
        resume_from_file = None 
    ),
    optim = dict(
        type="Adam",
        lr = 1e-4, #3e-4, 
        weight_decay = 0.0,
        beta1 = 0.9,
        beta2 = 0.999,
        cos = False,

        # weight_decay = 0.
    ),
    # optim = dict(
    #     type="SGD",
    #     lr = 0.05,
    #     momentum = 0.9,
    #     weight_decay = 2e-4
    # ),
    loss =  dict(
        type = "CrossEntropy"
    ),
    mode = {
        "train": 10,
        "eval": 1
    }
)

log_config = dict(
    log_level = "info",
    log_dir = None,
    when = 'D',
    backCount = 3
)