import os

import torch
import torch.multiprocessing as mp

def init_dist(cfg, backend="nccl", log=None, **kwargs):
    """ initialization for distributed training"""

    distributed = cfg.is_distributed
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) >= 1
    if distributed:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend=backend, **kwargs)                        
        cfg.gpus = torch.distributed.get_world_size()

    log.logger.info("Distributed training: {}".format(distributed))
    log.logger.info(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    log.logger.info("use distributed {}".format(distributed))


if __name__ == "__main__":

    torch.distributed.init_process_group('nccl')
    import time
    time.sleep(30)

    torch.distributed.destroy_process_group()
    # if torch.distributed.is_available():
    #     torch.distributed.init_process_group()