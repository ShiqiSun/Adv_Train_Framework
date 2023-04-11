device = 1
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0
# clamp -2.5 -2.5

model = dict(
    type = "WideResNet",
    depth = 28,
    num_classes = 10,
    widen_factor = 10,
    dropRate = None,
    stride_list = [2, 2, 2, 2]
)

dataset = dict(
    type = "ImageNette",   
    root_dir = None,
    train = dict(
        batch_size = 80,
    ),
    test = dict(
        batch_size = 80,
    ),
    normalize = dict(
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 1500,
    save_freq = 20,
    epoch = 300,
    lr_adjust_list = [50, 70, 90, 110],
    resume = dict(
        is_resume = True,
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
        type = "Trades",
        step_size = 0.003,
        epsilon = 0.138,
        perturb_steps = 20,
        beta = 1.0,
        dis_type = 'l_inf'
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