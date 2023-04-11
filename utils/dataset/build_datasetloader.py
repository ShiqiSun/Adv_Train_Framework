
from utils.register.registers import DATASETS
import utils.dataset.dataset_registers


def build_dataloader(cfg, log=None):

    data_loader = DATASETS[cfg.dataset.type](cfg)
    
    if cfg.local_rank == 0 or cfg.is_distributed == False:
        log.logger.info("Dataset:{}. Train Batch_size:{}. Test Batch_size:{}.".format( \
            cfg.dataset.type, cfg.dataset.train.batch_size, cfg.dataset.test.batch_size
        ))

    return data_loader

