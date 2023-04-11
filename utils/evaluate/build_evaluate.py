from utils.train.trainer import Trainer
from utils.dataset.build_datasetloader import build_dataloader
from utils.models.build_models import build_classifier
from utils.attack.build_attacker import build_attacker

def evaluate(cfg, load_file=None, log=None, is_trainset=False):
    model = build_classifier(cfg, log=log)

    data_loader = build_dataloader(cfg, log=log)

    if load_file == None:
        cfg.train.resume.is_resume = True
        cfg.train.resume.resume_from_work_dir = True
    elif load_file == "pretrain":
        load_file = None
    else:
        cfg.train.resume.is_resume = True
        cfg.train.resume.resume_from_work_dir = False
        cfg.train.resume.resume_from_file = load_file
    
    cfg.gpus = 1

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        cfg=cfg,
        log=log
    )

    trainer.eval(train_eval=is_trainset)

def evaluate_attack(cfg, load_file=None, log=None, is_trainset=False, cfg_atk=None):

    model = build_classifier(cfg, log=log)

    attacker = build_attacker(cfg_atk.type, log=log)

    data_loader = build_dataloader(cfg, log=log)

    log.logger.info(f"Attack cfg is {cfg_atk}")

    if load_file == None:
        cfg.train.resume.is_resume = True
        cfg.train.resume.resume_from_work_dir = True
    elif load_file == "pretrain":
        load_file = None
    else:
        cfg.train.resume.is_resume = True
        cfg.train.resume.resume_from_work_dir = False
        cfg.train.resume.resume_from_file = load_file
    
    cfg.gpus = 1

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        cfg=cfg,
        log=log
    )


    trainer.eval_from_atk(attacker, cfg_atk, train_eval=is_trainset)