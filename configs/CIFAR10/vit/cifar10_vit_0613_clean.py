from unicodedata import is_normalized


device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = True
local_rank = 0

model = dict(
    type = "ViT",
    image_size=32, 
    patch_size=4, 
    num_classes=10, 
    dim=512, #256
    depth=6, #6
    heads=8, #8
    mlp_dim=512, #512
    dropout = 0.1, 
    emb_dropout = 0.1,
    # generally not change
    # pool = 'mean',
    pool = 'cls', 
    channels = 3, 
    dim_head = 64, 
)

dataset = dict(
    type = "CIFAR10",  
    # image_size = 256, 
    root_dir = None,
    train = dict(
        batch_size = 128,
    ),
    test = dict(
        batch_size = 128,
    ),
    normalize = dict(
        is_normalized = True,
        mean = None,
        std = None
    )
)

train = dict(
    log_interval = 40,
    save_freq = 20,
    epoch = 300,
    lr_adjust_list = [20, 40, 60, 80], #[30, 50, 70, 90]
    resume = dict(
        is_resume = False,
        resume_from_work_dir = False,
        resume_from_file = None 
    ),
    optim = dict(
        type="Adam",
        lr = 1e-3, #3e-4, 
        weight_decay = 5e-5,
        beta1 = 0.9,
        beta2 = 0.999,
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