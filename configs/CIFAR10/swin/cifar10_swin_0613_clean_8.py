device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = False
local_rank = 0

model = dict(
    type = "swin_t",
    patch_size=4, 
    num_classes=10, 
    downscaling_factors=(2,2,2,1)
)

dataset = dict(
    type = "CIFAR10",  
    # image_size = 256, 
    root_dir = None,
    train = dict(
        batch_size = 512,
    ),
    test = dict(
        batch_size = 512,
    ),
    normalize = dict(
        is_normalized = True,
        mean = [0.4914, 0.4822, 0.4465],
        std = [0.2023, 0.1994, 0.2010]
    )
)

train = dict(
    log_interval = 40,
    save_freq = 20,
    epoch = 300,
    lr_adjust_list = [100, 140, 160, 180], #[30, 50, 70, 90]
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
    #     lr = 0.1,
    #     momentum = 0.9,
    #     weight_decay = 2e-4
    # ),
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