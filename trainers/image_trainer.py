from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import submitit
import subprocess
import hashlib
from torch.optim import Adam, SGD
import torch.nn as nn
import torch

from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import get_original_cwd
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import os
import logging
from abc import abstractmethod
from torchinfo import summary


import data.dataloaders as dl
from models import model_utils as mut
from trainers import trainer_utils as tut
from trainers.mixed_trainer import MultiLossTrainer
from data.masking import Mask_Manipulator

LOG = logging.getLogger(__name__)

''' NOTE: Now that outputs and labels are dict, it would be possible to refactor many of these trainers into single a single class - however, this could make the code less readable'''

class ImCorr_trainer(MultiLossTrainer):
    ''' This inference takes a Model composed of Image and DCM modules. Thus forward needs X,Z and images'''
    ''' Z variable is not used in our experiments, but is kept for consistency with other trainers'''

    def __init__(self, cfg):
        super(ImCorr_trainer, self).__init__(cfg)

    def setup_optimizer(self):
        '''The utility layer is learnt with higher LR, due to the different order of magnitude needed for beta parameters '''
        ''' The utility layer will not have any weight decay'''
        utility_named = 'utility'  # WARNING: any layer containing the name utility will have a different learning rate
        utility_params = []
        NN_params = []
        for name, parameter in self.model.named_parameters():
            if utility_named in name:
                utility_params.append(parameter)
            else:
                NN_params.append(parameter)
        if self.cfg.trainer.optimizer == 'Adam':
            return Adam([
                    {'params': utility_params,
                     'lr': self.cfg.trainer.param.lr*self.cfg.trainer.param.utility_lr_multiplier},
                    {'params': NN_params, 'lr': self.cfg.trainer.param.lr,
                     'weight_decay': self.cfg.trainer.param.weight_decay}
                    ])
        elif self.cfg.trainer.optimizer == 'SGD':
            return SGD([
                    {'params': utility_params,
                     'lr': self.cfg.trainer.param.lr*self.cfg.trainer.param.utility_lr_multiplier, 'momentum': self.cfg.trainer.param.momentum},
                    {'params': NN_params, 'lr': self.cfg.trainer.param.lr,
                     'weight_decay': self.cfg.trainer.param.weight_decay, 'momentum': self.cfg.trainer.param.momentum}
                    ])

    def infer_model(self, input):
        X, Z, labels, images = self.process_inputs(input)
        outputs = self.model(X, images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice': labels} , size

    def process_inputs(self, input):
        X, Z, labels, images = input
        X = X.to(self.train_device)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X) # Masking is done here for tabular data, not in dataloader because we often need original data
        return (X, Z, labels, images)

class ImChoice_trainer(MultiLossTrainer):
    ''' Simply infers choice based on Image '''

    def __init__(self, cfg):
        super(ImChoice_trainer, self).__init__(cfg)

    def infer_model(self, input):
        X, Z, labels, images = self.process_inputs(input)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        outputs = self.model(images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice': labels}, size

    def process_inputs(self, input):
        X, Z, labels, images = input
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        return (X, Z, labels, images)
    
class Detection_trainer(MultiLossTrainer):
    ''' Labels are X for detection '''

    def __init__(self, cfg):
        super(Detection_trainer, self).__init__(cfg)

    def infer_model(self, input):
        "The labels are replaced by X => we want to detect the ground truths that generated the image"
        X, Z, labels, images = self.process_inputs(input)
        outputs = self.model(images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'X': X.clone().detach()}, size

    def process_inputs(self, input):
        X, Z, labels, images = input
        X = X.to(self.train_device)
        Z = Z.to(self.train_device)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X)
        return (X, Z, labels, images)

    def eval(self):
        LOG.info(
            'Eval not yet implemented for Detection Model (though not necessary if no change in dataset generation)')
        return 0

class GAIN_trainer(Detection_trainer):
    ''' Same inference as ChoiceDetect_Corr, but inherit detection class'''
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def infer_model(self, input):
        X, Z, labels, images = self.process_inputs(input)
        X_target = X.clone().detach()
        outputs = self.model(X, images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'X': X_target}, size
    
    def process_inputs(self, input):
        X, Z, labels, images = input
        X = X.to(self.train_device)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X)
        return (X, Z, labels, images)


class Choice_Detection_trainer(MultiLossTrainer):
    ''' Inference require labels for choice & detection '''

    def __init__(self, cfg):
        super(Choice_Detection_trainer, self).__init__(cfg)

    def infer_model(self, input):
        ''' Outputs contain main and subtask, Labels are returned with same list order '''
        X, Z, labels, images = self.process_inputs(input)
        outputs = self.model(images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice':labels, 'X': X.clone().detach()}, size
    
    def process_inputs(self, input):
        X, Z, labels, images = input
        X = X.to(self.train_device)
        Z = Z.to(self.train_device)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X)
        return (X, Z, labels, images)

class Choice_Detection_Corr_trainer(ImCorr_trainer):
    ''' The models requires all X, Z and Images + Output 2 types of labels'''

    def infer_model(self, input):
        X, Z, labels, images = self.process_inputs(input)
        X_target = X.clone().detach()
        outputs = self.model(X, images)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice':labels, 'X': X_target}, size
    
    def process_inputs(self, input):
        X, Z, labels, images = input
        X = X.to(self.train_device)
        images = images.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X)
        return (X, Z, labels, images)
    
# class Contrastive_trainer(MultiLossTrainer):
#     ''' Simply infers choice based on Image '''

#     def __init__(self, cfg):
#         super(Contrastive_trainer, self).__init__(cfg)

#     def infer_model(self, input):
#         X, Z, labels, images = input
#         X = X.to(self.train_device)
#         images = images.to(self.train_device)
#         labels = labels.to(self.train_device)
#         X = self.masker.mask_tabular(X)
#         X_target = X.clone().detach()
#         outputs = self.model(images, X)
#         size = torch.tensor(len(labels)).float().to(self.train_device)
#         return outputs, {'choice':labels, 'X': X_target}, size