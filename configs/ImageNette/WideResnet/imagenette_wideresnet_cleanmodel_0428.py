device = 1
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = False
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
    type = "ImageNette",   
    root_dir = None,
    train = dict(
        batch_size = 256,
    ),
    test = dict(
        batch_size = 256,
    ),
    normalize = dict(
        mean = [0.5, 0.5, 0.5],
        std = [0.5, 0.5, 0.5]
    )
)

train = dict(
    log_interval = 1500,
    save_freq = 30,
    epoch = 300,
    lr_adjust_list = [50, 70, 90, 110],
    resume = dict(
        is_resume = False,
        resume_from_work_dir = True,
        resume_from_file = None 
    ),
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
        "train": 10,
        "eval": 1
    }
)

log_config = dict(
    log_level = "info",
    log_dir = None,
    when = 'D',
    backCount = 10
)