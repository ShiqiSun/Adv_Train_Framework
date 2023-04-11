
from utils.train.trainer import Trainer
from utils.dataset.build_datasetloader import build_dataloader
from utils.models.build_models import build_classifier


def train_classifier(cfg, log=None):

    model = build_classifier(cfg, log=log)

    data_loader = build_dataloader(cfg, log=log)

    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        cfg=cfg,
        log=log
    )

    trainer.run()