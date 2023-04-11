device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])

is_distributed = False

model = dict(
    type = "FFN",
    layers = 2,
)

dataset = dict(
    type = "MNIST",   
    root_dir = None,
    train = dict(
        batch_size = 512,
    ),
    test = dict(
        batch_size = 512,
    ),
)

train = dict(
    log_interval = 40,
    save_freq = 10,
    epoch = 300,
    lr_adjust_list = [30, 50, 70, 90],
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