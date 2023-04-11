#clamp 0, 1 to test result 
device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0

model = dict(
    type = "WideResNet",
    depth = 28,
    num_classes = 10,
    widen_factor = 10,
    dropRate = None,
    stride_list = [1, 1, 2, 2]
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
        is_normalized = True,
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 750,
    save_freq = 10,
    epoch = 200,
    lr_adjust_list = [30, 50, 70, 90],
    resume = dict(
        is_resume = False,
        resume_from_work_dir = False,
        resume_from_file = None 
    ),
    optim = dict(
        type="SGD",
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 2e-4
    ),
    loss =  dict(
        type = "Adv",
        epsilon = 0.138,
        perturb_steps = 20,
        step_size = 0.0138
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