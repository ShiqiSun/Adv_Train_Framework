device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0

model = dict(
    type = "ViT",
    image_size=32, 
    patch_size=4, 
    num_classes=10, 
    dim=256, 
    depth=6, 
    heads=16, 
    mlp_dim=512, 
    pool = 'cls', 
    channels = 3, 
    dim_head = 64, 
    dropout = 0.1, 
    emb_dropout = 0.1
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
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0]
    )
)

train = dict(
    log_interval = 40,
    save_freq = 10,
    epoch = 300,
    lr_adjust_list = [30, 50, 70, 90],
    resume = dict(
        is_resume = False,
        resume_from_work_dir = False,
        resume_from_file = None 
    ),
    # optim = dict(
    #     type="Adam",
    #     lr = 0.1,
    # ),
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
    backCount = 3
)