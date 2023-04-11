import argparse
import sys

sys.path.append("./")

from utils.fileio.config import Config
from utils.log.log import get_logger
from utils.evaluate.build_evaluate import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a Classifier")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--model", default=None, help="evaluate from the model")
    parser.add_argument("--is_trainset", default=False, type=bool, help="evaluate from the model")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.is_distributed  = False
    log = get_logger(cfg)

    evaluate(cfg=cfg, load_file=args.model, log=log, is_trainset=args.is_trainset)
    

if __name__ == "__main__":
    main()