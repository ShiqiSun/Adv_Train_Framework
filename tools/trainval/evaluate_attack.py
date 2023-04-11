import argparse
import sys

sys.path.append("./")

from utils.fileio.config import Config
from utils.log.log import get_logger
from utils.evaluate.build_evaluate import evaluate_attack

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate From Attack")
    parser.add_argument("--config",
                        default="configs/adv_config/lipKLJab_config/imagette_lipKLJab_0510.py" ,
                        # default="configs/adv_config/trades_config/imagette_trades_0509.py" ,
                        # default="configs/ImageNette/WideResnet/imagenette_wideresnet_cleanmodel_0510.py",
                        help="train config file path")
    parser.add_argument("--model", default=None, help="evaluate from the model")
    parser.add_argument("--is_trainset", default=False, type=bool, help="evaluate from the model")
    parser.add_argument("--cfg_atk", default="configs/Attack/pgd_whitbox_imagenette.py", type=str, help="evaluate from the model")
    parser.add_argument("--test_batchsize", default=300, type=int, help="evaluate batchsize")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.device = 1
    cfg.local_rank = cfg.device
    print(f"device is {cfg.device}")
    cfg.is_distributed  = False
    cfg_atk = Config.fromfile(args.cfg_atk)

    cfg.dataset.test.batch_size = args.test_batchsize
    log = get_logger(cfg)
    evaluate_attack(cfg=cfg, load_file=args.model, log=log, is_trainset=args.is_trainset, cfg_atk=cfg_atk)
    

if __name__ == "__main__":
    main()