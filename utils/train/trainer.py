import os
import os.path as osp

import torch
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel

from utils.fileio.misc import is_str
from utils.fileio.path import mkdir_or_exist
from utils.fileio.io import find_new_file_with_ext
from utils.optimizer.build_optimizer import build_optimizer
from utils.loss.build_loss import build_loss
from utils.evaluate.evaluate_train import eval_from_dataloader
from utils.evaluate.evaluate_attack import eval_atk_from_dataset

class Trainer(object):
    def __init__(self,
                model,
                data_loader,
                cfg,
                log=None,
                **kwargs,) -> None:
        
        self._model = model
        self._data_loader = data_loader
        self._cfg = cfg
        self._device = cfg.device
        self._epoch = 1
        self._max_epoch = cfg.train.epoch
        self._lr_adjust_list = cfg.train.lr_adjust_list
        self._lr = cfg.train.optim.lr
        self._mode = cfg.train.mode
        self._save_freq = cfg.train.save_freq
        self._log = log
        self._local_rank = cfg.local_rank
        self._is_distributed = cfg.is_distributed

        work_dir = cfg.work_dir
        if is_str(work_dir):
            self._work_dir = osp.abspath(work_dir)
            mkdir_or_exist(self._work_dir)
        else:
            raise TypeError("'work_dir' must be a str")

        if "resume" in self._cfg.train:
            self.resume(self._cfg.train.resume)

        if self._cfg.is_distributed:
            self.model_dist()
        
    
    
    def model_dist(self):
        if self._local_rank == 0:
            self._log.logger.info("Distributed Training Start.")
        self._model = SyncBatchNorm.convert_sync_batchnorm(self._model)
        self._model = DistributedDataParallel(
                self._model, 
                device_ids=[self._device],
                output_device=self._device, 
                find_unused_parameters=True, #Todo
                broadcast_buffers=False, #Todo see how it works
                )

    def resume(self, cfg):
        # is_resume > resume_from_work_dir > resume_from_file
        if cfg.is_resume:
            if cfg.resume_from_work_dir or cfg.resume_from_file ==None:
                load_file = find_new_file_with_ext(self._work_dir, 'pt')
                print(load_file)
            elif cfg.resume_from_file != None:
                load_file = cfg.resume_from_file
            if load_file == None and (not self._is_distributed or self._local_rank == 0):
                self._log.logger.info("Find No Load file. Will Start From Init Point.")
            else:
                # self._model.load_state_dict(torch.load(load_file))
                self._model.load_state_dict({k.replace('module.', ''): v for k, v in                 
                                    torch.load(load_file).items()})
                if (not self._is_distributed or self._local_rank == 0):
                    self._log.logger.info("Successfully Load Resume Model From {}".format(load_file))

    def init_optimizer(self):
        optimizer = build_optimizer(self._cfg.train.optim, self._model)
        if (not self._is_distributed or self._local_rank == 0):
            self._log.logger.info("Optimizer:{}".format(self._cfg.train.optim))
        return optimizer

    def init_loss(self):
        loss = build_loss(self._cfg.train.loss)
        if (not self._is_distributed or self._local_rank == 0):
            self._log.logger.info("Loss:{}".format(self._cfg.train.loss))
        return loss
    
    def init_scheduler(self):
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(self._optim, 'min', 
                            patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
        return scheduler

    def train(self):
        self._model.train()
        self.adjust_learning_rate()
        train_loader = self._data_loader["train"]
        if (not self._is_distributed or self._local_rank == 0):
            self._log.logger.info("=========================Epoch{}=========================".format(self._epoch))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self._device), target.to(device=self._device)
            self._optim.zero_grad()
            loss = self._loss(self._model, data, target, self._cfg, self._optim)
            loss.backward()
            self._optim.step()
            if batch_idx % self._cfg.train.log_interval == 0 and (not self._is_distributed or self._local_rank == 0):
                self._log.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self._epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        
        if self._epoch % self._save_freq == 0 and (not self._is_distributed or self._local_rank == 0):
            self.save_model()
        self._epoch += 1

    def save_model(self):
        self._log.logger.info("=========================Save=========================")
        self._log.logger.info("Save model at epoch {}".format(self._epoch))
        if self._is_distributed:
            torch.save(self._model.module.state_dict(),
                        os.path.join(self._work_dir, 'model-nn-epoch{}.pt'.format(self._epoch)))
        else:
             torch.save(self._model.state_dict(),
                        os.path.join(self._work_dir, 'model-nn-epoch{}.pt'.format(self._epoch)))
        torch.save(self._optim.state_dict(),
                   os.path.join(self._work_dir, 'opt-nn-checkpoint_epoch{}.tar'.format(self._epoch)))

    def eval(self, train_eval=False):
        if (not self._is_distributed or self._local_rank == 0):
            self._log.logger.info("=========================Evaluate=========================")
        self._model.eval()
        _, _, info_test = eval_from_dataloader(self._model, self._device, self._data_loader["test"])
        if train_eval:
            _, _, info_train = eval_from_dataloader(self._model, self._device, self._data_loader["train"], gpus=self._cfg.gpus)
        if (not self._is_distributed or self._local_rank == 0):
            if train_eval:
                self._log.logger.info("Training: {}".format(info_train))
            self._log.logger.info("Testing: {}".format(info_test))

    def run(self):
        self._optim = self.init_optimizer()
        self._loss = self.init_loss()
        if self._cfg.train.optim.type == "Adam" and self._cfg.train.optim.cos: 
            self._scheduler = self.init_scheduler()
        while self._epoch <= self._max_epoch:
            for mode, times in self._mode.items():
                for _ in range(times):
                    self.call_hook(mode)
                continue
        if (not self._is_distributed or self._local_rank == 0):
            self._log.logger.info("Running Finish!")

    def call_hook(self, fn_name):
        getattr(self, fn_name)()

    def adjust_learning_rate(self):
        if self._cfg.train.optim.type == "Adam" and self._cfg.train.optim.cos:
            self._scheduler.step(self._epoch)
        else:
            lr_t = self._lr
            lr = lr_t
            if self._epoch >= self._lr_adjust_list[0]:
                lr = lr_t * 0.1
            if self._epoch >= self._lr_adjust_list[1]:
                lr = lr_t * 0.01
            if self._epoch >= self._lr_adjust_list[2]:
                lr = lr_t * 0.001
            for param_group in self._optim.param_groups:
                param_group['lr'] = lr


    def eval_from_atk(self, attacker, cfg_atk, train_eval=False):
        self._log.logger.info(f"Attack Evaluate Start:{cfg_atk}")        
        if train_eval:
            self._log.logger.info(f"Trainset Evaluate:")
            eval_atk_from_dataset(self._model,
                self._data_loader["train"],
                self._device, 
                attacker,
                cfg_atk,
                self._log
                )
        self._log.logger.info(f"Testset Evaluate:")
        eval_atk_from_dataset(self._model,
                self._data_loader["test"],
                self._device, 
                attacker,
                cfg_atk,
                self._log
                )  
        return
