device = 0
work_dir = '../work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
is_distributed = False
local_rank = 0

model = dict(
    type = "WideResNet"
)