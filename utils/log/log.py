from fileinput import filename
import os
import os.path as osp

import logging
from logging import handlers

from utils.fileio.misc import is_str
from utils.fileio.path import mkdir_or_exist

class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'critical':logging.CRITICAL
    }

    def __init__(self,
                log_dir,
                log_level="info",
                when='D',
                backCount=3,
                fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                ) -> None:
        """
        log_levl:output which level
        filename: log save path
        fmt: output style of log
        when:interval of saving log
            S(econd) M(inute) H(our) D(ay) W(eek) midnight
        """
        if is_str(log_dir):
            log_dir = osp.abspath(log_dir)
            mkdir_or_exist(log_dir)
        else:
            raise TypeError("log path must be a str")
        
        filename = os.path.join(log_dir, log_level+'.log')

        log_level = log_level.lower()
        self.logger = logging.getLogger() #initize log
        format_str = logging.Formatter(fmt) # set log style
        self.logger.setLevel(self.level_relations.get(log_level)) # set log level
        
        sh = logging.StreamHandler() #output terminal
        sh.setFormatter(format_str)

        th = handlers.TimedRotatingFileHandler(filename=filename, \
                when=when,backupCount=backCount,encoding='utf-8')
        th.setFormatter(format_str) 

        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def get_logger(cfg):
    if "log_config" in cfg: 
        if cfg.log_config.log_dir == None:
            log_dir = os.path.join(cfg.work_dir, "log")
        else:
            log_dir = cfg.log_config.log_dir 
        log_level = cfg.log_config.log_level
        when=cfg.log_config.when
        backCount=cfg.log_config.backCount
    else:
        log_dir = os.path.join(cfg.work_dir, "log")
        log_level="info"
        when="D"
        backCount=3

    log = Logger(
        log_dir=log_dir,
        log_level=log_level,
        when=when,
        backCount=backCount
    )

    log.logger.info("Log file will be saved at {}".format(log_dir))

    return log

if __name__ == "__main__":
    log = Logger('test1.log', log_level="INFO")
    log.logger.info("test1")
    log.logger.info("test2")
    log.logger.info("test3")
    log.logger.info("test4")