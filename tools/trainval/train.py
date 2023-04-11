import argparse
from mimetypes import init
import sys

sys.path.append("./")

import os

from utils.fileio.config import Config
from utils.train.train import train_classifier
from utils.log.log import get_logger
from utils.dist.dist import init_dist

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Classifier")
    parser.add_argument("--config", default="configs/test_config2.py",help="train config file path")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    log = get_logger(cfg)

    if cfg.is_distributed:
        cfg.local_rank = args.local_rank
        cfg.device = cfg.local_rank
        init_dist(cfg, log=log)
    else:
        cfg.local_rank = cfg.device
        cfg.gpus = 1

    train_classifier(cfg, log=log)

if __name__ == "__main__":
    main()