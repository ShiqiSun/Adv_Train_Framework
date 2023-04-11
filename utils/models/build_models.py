from utils.register.registers import MODELS
import utils.models.classifier.classifiers_register

def build_classifier(cfg, log=None):

    model = MODELS[cfg.model.type](cfg.model).to(cfg.device)

    if cfg.local_rank == 0 or cfg.is_distributed == False:
        log.logger.info("Model {} is built:".format(cfg.model.type))
        log.logger.info(cfg.model)
    
    return model