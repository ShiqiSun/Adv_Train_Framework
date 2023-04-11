device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0

model = dict(
    type = "ResNet50",
    pretrained = True,
)


dataset = dict(
    type = "ImageNet",   
    root_dir = None,
    train = dict(
        batch_size = 256,
    ),
    test = dict(
        batch_size = 1024,
    ),
    normalize = dict(
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 750,
    save_freq = 5,
    epoch = 300,
    lr_adjust_list = [10, 30, 50, 70],
    resume = dict(
        is_resume = False,
        resume_from_work_dir = True,
        resume_from_file = None
    ),
    optim = dict(
        type="SGD",
        lr = 0.05,
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
    backCount = 10 
)