from torch.utils.tensorboard import SummaryWriter
import time
from tqdm import tqdm
import submitit
import subprocess
import hashlib
from torch.optim import Adam, SGD
import torch.nn as nn
import torch

from omegaconf import DictConfig, OmegaConf, open_dict, ListConfig
from hydra.utils import get_original_cwd, to_absolute_path
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import sys
import os
import json
import logging
import pickle
from abc import abstractmethod
from torchinfo import summary

import data.dataloaders as dl
from data.masking import Mask_Manipulator
from models import model_utils as mut
from trainers import trainer_utils as tut
from trainers import eval_utils as eut

LOG = logging.getLogger(__name__)

class BaseMultiTrainer(object):
    ''' Base class for all trainers.  Contains all the boilerplate code for training, logging, checkpointing, etc.'''
    current_epoch = 0
    model = None
    optimizer = None

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def setup_local(self) -> None:
        using_cuda = torch.cuda.is_available() and self.cfg.trainer.cuda
        if (not using_cuda) and self.cfg.trainer.cuda:
            LOG.warning('CUDA is not available, using CPU')
        self.train_device = torch.device('cuda' if using_cuda else 'cpu')
        # Hydra does a chdir, so fixing paths:
        chk_paths = self.cfg.trainer.checkpointpath
        ''' Turn to list, so models can have 2 checkpoints '''
        if not (isinstance(chk_paths, list) or isinstance(chk_paths, ListConfig)):
            chk_paths = [chk_paths]
        self.cfg.trainer.checkpointpath = [to_absolute_path(chk_path) for chk_path in chk_paths]
        model_info_path = to_absolute_path(self.cfg.trainer.model_registry)
        if os.path.exists(model_info_path):
            LOG.info('Model directory avaialable at {}'.format(model_info_path))
        else:
            models = {}
            with open(model_info_path, 'w') as f:
               json.dump(models, f)
            LOG.info('Model directory created at {}'.format(model_info_path))

        self.setup_extras()

    @abstractmethod
    def setup_extras(self) -> None:
        '''Code specific - override in child class'''
        pass

    def setup_slurm(self) -> None:
        # Changed manual submitit to Hydra's launcher, setup is currently the same
        self.setup_local()

    def run(self) -> None:
        try:
            self.train()
        except KeyboardInterrupt:
            self.checkpoint_dump(epoch=self.current_epoch)
            raise

    def setup_trainer(self) -> None:
        LOG.info(
            f"Survey trainer: {self.cfg.trainer.rank}, device: {self.train_device}")

        # Setup logging    
        if self.cfg.trainer.use_clearml:
            from clearml import Task
            task = Task.init(
                    project_name=self.cfg.project_name, task_name=self.cfg.trainer.ml_exp_name
                )
            task.connect(OmegaConf.to_container(self.cfg, resolve=False))
        if self.cfg.trainer.use_wandb:
            import wandb
            wandb.tensorboard.patch(root_logdir='runs', pytorch=True)
            wandb.init(
                    project=self.cfg.project_name, name=self.cfg.trainer.name,
                    config= OmegaConf.to_container(self.cfg, resolve=False)
                )
        self.writer = None
        if self.cfg.trainer.use_tensorboard:
            # write logs within hydra working folder
            self.writer = SummaryWriter(log_dir='runs')

        LOG.info(
            f"Getting Dataloaders")
        time_start = time.time()
        
        # Get DATALOADERS and masking class
        [self.train_dataloader, self.val_dataloader, self.test_dataloader], self.masker = dl.getDataloaders(
            self.cfg)
        LOG.info("Done loading data in {}".format(time.time()-time_start))
        
        # Instantiate the model and optimizer
        self.model = mut.get_model(self.cfg)

        LOG.info('Model Loaded: \n')
        summary(self.model)
        self.criterion = tut.get_loss(self.cfg.trainer.loss, tut.create_lambda_dict(self.cfg.trainer.param))
        self.optimizer = self.setup_optimizer()
        self.current_epoch = 0
        self.model.to(self.train_device)

        ######## Looking for checkpoints
        LOG.info('Looking for existing checkpoints in output dir...')
        chk_paths = self.cfg.trainer.checkpointpath
        ''' Turn to list, so 2 models can have a checkpoint '''
        if not (isinstance(chk_paths, list) or isinstance(chk_paths, ListConfig)):
            chk_paths = [chk_paths]
        for chk_path in chk_paths:
            ''' Load models from checkpoint path'''
            self.custom_checkpoint_load(chk_path)

        if self.cfg.trainer.load_model_class:
            ''' Load model from a json registry - ! tested but unverified Functionality '''
            model_info_path = to_absolute_path(self.cfg.trainer.model_registry)
            with open(model_info_path, 'r') as f:
                models = json.load(f)
            if self.cfg.trainer.load_model_class in models.keys():
                if self.cfg.trainer.load_best_model:
                    chk_path = models[self.cfg.trainer.load_model_class]['best']['model_path']
                    LOG.info('Loading best model from {}'.format(chk_path))
                else:
                    chk_path = models[self.cfg.trainer.load_model_class]['last']['model_path']
                self.custom_checkpoint_load(chk_path)
            else:
                LOG.error('Model {} not found in registry'.format(self.cfg.trainer.load_model_class))


        self.optimizer.zero_grad()
        LOG.info("Initialization passed successfully.")

    def setup_optimizer(self):
        ''' Adam by default.  Made a function for easier class overrides '''
        if self.cfg.trainer.optimizer == 'Adam':
            return Adam(self.model.parameters(),
                    lr=self.cfg.trainer.param.lr,
                    weight_decay=self.cfg.trainer.param.weight_decay)
        elif self.cfg.trainer.optimizer == 'SGD':
            return torch.optim.SGD(self.model.parameters(),
                    lr=self.cfg.trainer.param.lr,
                    momentum=self.cfg.trainer.param.momentum,
                    weight_decay=self.cfg.trainer.param.weight_decay)
        else:
            raise NotImplementedError(
                "Unknown optimizer {} (valid values are Adam or SGD)".format(self.cfg.trainer.optimizer))

    @ abstractmethod
    def train(self) -> None:
        pass

    @ abstractmethod
    def eval(self) -> None:
        pass

    @ abstractmethod
    def model_hook(self) -> None:
        pass

    def setup_platform(self) -> None:
        tut.fix_random_seeds(self.cfg.trainer.seed)
        if self.cfg.trainer.platform == "local":
            LOG.info(f"Training platform : {self.cfg.trainer.platform}")
            self.setup_local()
        elif self.cfg.trainer.platform == "slurm":
            LOG.info(f"Training platform : {self.cfg.trainer.platform}")
            self.setup_slurm()
        else:
            raise NotImplementedError(
                "Unknown platform (valid value are local or slurm)")

    def __call__(self) -> None:
        self.setup_platform()
        self.setup_trainer()
        self.run()
        self.eval()

    def log_iteration(self, iteration, epoch, metrics):
        """ Prints an iteration to Console showing the current loss, saves metrics to writer.
        :param int iteration:  how many batches seen in this epoch
        :param int epoch:  current epoch
        :param dict metrics: a dictionary containing all metrics of a single step . Losses and Scores are also dictionaries (multi)
        """
        if iteration % self.cfg.trainer.log.every_n_iter == 0:
            losses, accs, size = metrics['losses'], metrics['scores'], metrics['size']
            list_loss = []
            list_acc = []
            if self.writer is not None:
                for loss_name, loss in losses.items():
                    self.writer.add_scalar(
                        f"Loss/Train_{loss_name}", loss, epoch*len(self.train_dataloader)+iteration)
                    list_loss.append(loss)
                for acc_name, acc in accs.items():
                    self.writer.add_scalar(
                        f"Acc/Train_{acc_name}", acc/size, epoch*len(self.train_dataloader)+iteration)
                    list_acc.append(acc/size)
            LOG.info(
                f"{self.cfg.trainer.rank}:{self.train_device} - losses: {list_loss}\t accuracies: {list_acc}")

    def log_epoch(self, epoch, iteration, step_dicts, subset='Train'):
        ''' step_dicts is a list of metrics, 1 per iteration. See log_iteration '''
        # [{scores: {det: x, choice: u}, loss: ...}, {}, ...]
        scores = [x['scores'] for x in step_dicts]
        scores_types = scores[0].keys()
        losses = [x['losses'] for x in step_dicts]
        losses_types = losses[0].keys()

        sizes = torch.stack([x['size'] for x in step_dicts])
        total_size = sizes.sum()
        epoch_logs = {}

        LOG.info(
            f"{self.cfg.trainer.rank}:{self.train_device} - epoch {subset}:")
        for loss_type in losses_types:
            loss = torch.stack([x[loss_type] for x in losses])
            avg_loss = loss@sizes/total_size
            epoch_logs.update({f"Loss/{subset}_{loss_type}": avg_loss})
            if self.writer is not None:
                self.writer.add_scalar(f"Loss/{subset}_{loss_type}", avg_loss, (epoch+1)*iteration)
            LOG.info(f"-------- Loss_{loss_type}: {avg_loss}")
        for score_type in scores_types:
            score = torch.stack([x[score_type] for x in scores]).sum()
            avg_acc = score/float(total_size)
            epoch_logs.update({f"Acc/{subset}_{score_type}": avg_acc})
            if self.writer is not None:
                self.writer.add_scalar(f"Acc/{subset}_{score_type}", avg_acc, (epoch+1)*iteration)
            LOG.info(f"-------- Acc_{score_type}: {avg_acc}")

        return epoch_logs

    #-------------------------------------
    # https://github.com/facebookincubator/submitit/blob/main/docs/checkpointing.md

    def checkpoint_dump(
        self,
        checkpoint_path: str = None,
        epoch: int = 0,
        model_state_dict: Dict = None,
        optimizer_state_dict: Dict = None,
        **kwargs,
    ) -> None:
        if (model_state_dict is None) and (self.model is not None):
            model_state_dict = self.model.state_dict()

        if (optimizer_state_dict is None) and (self.optimizer is not None):
            optimizer_state_dict = self.optimizer.state_dict()

        if checkpoint_path is None:
            prefix = '' # should still work with slurm since hydra run or sweep dir is same disk/location as output_dir anyway. 
            ## Old code from mixed up checkpointing system: 
            # prefix = self.cfg.trainer.output_dir
            # if self.cfg.trainer.platform == "local":  # Hydra changes base directory
            #     prefix = ''
            if epoch == 0:
                checkpoint_path = os.path.join(prefix, "default_checkpoint.pt")
            else:
                checkpoint_path = os.path.join(
                    prefix, f"checkpoint_epoch_{str(epoch)}.pt")

        self.cfg.trainer.checkpointpath_save = os.path.abspath(os.path.join(os.getcwd(),checkpoint_path))

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                **kwargs
            },
            checkpoint_path
        )
        if self.cfg.trainer.use_wandb:
            import wandb
            ## Replacing artifact with model_path logging (text_log)
            # artifact = wandb.Artifact(
            #     name=self.cfg.trainer.ml_exp_name+str(epoch),
            #     type='model'
            #     )
            # artifact.add_reference('file://'+checkpoint_path)
            # wandb.log_artifact(artifact)
            text_log = wandb.Table(data=[[self.cfg.trainer.checkpointpath_save]], columns = ["model_path"])
            wandb.log({"model_path": text_log})


    def checkpoint_load(self, checkpoint_path: Union[str, Path]) -> Optional[Dict]:
        if not checkpoint_path:
            return None
        if not Path(checkpoint_path).exists():
            LOG.warning('No Checkpoints found at:  {}'.format(checkpoint_path))
            return None
        return torch.load(str(checkpoint_path), map_location=self.train_device)

    def find_latest_checkpoint_path(self, output_dir: Union[str, Path]) -> Optional[Path]:
        ''' Deprecated, using model json registry instead'''
        p = Path(output_dir)
        if not p.exists():
            return None
        checkpoints = [x for x in p.iterdir() if x.is_file()
                       and x.suffix == '.pt']
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: int(
            x.stem.split('_')[-1]), reverse=True)
        return checkpoints[0]

    def custom_checkpoint_load(self, chk_path):
        ''' Some model.load_state_dict have an override if they can load partial model components (e.g., DCM and image component seperately) '''
        checkpoint_dict = self.checkpoint_load(chk_path)
        if checkpoint_dict:
            # Load Model parameters
            LOG.info(f'Checkpoint found: {str(chk_path)}')
            # Load Model parameters
            # Composed models can do a partial loading
            custom_return = self.model.load_state_dict(
                checkpoint_dict["model_state_dict"])
            # Define and load parameters of the optimizer
            if custom_return is None:
                self.optimizer.load_state_dict(
                    checkpoint_dict["optimizer_state_dict"])
                # Track the epoch at which the training stopped
                self.current_epoch = checkpoint_dict["epoch"]
                # The following two lines load models which were not saved correctly: (Todo, Re-run useful models and clear lines)
                if self.current_epoch is None:
                    self.current_epoch = 0
            else:
                LOG.info(f'Custom loading occured, optimizer checkpoint cannot be loaded (known ML/pytorch limitation)')


    ###Copilot generated - unused but kept for reference:
    def checkpoint_load_latest(self, output_dir: Union[str, Path]) -> Optional[Dict]:
        checkpoint_path = self.find_latest_checkpoint_path(output_dir)
        if checkpoint_path is None:
            return None
        return self.checkpoint_load(checkpoint_path)

    def checkpoint_load_latest_and_resume(self, output_dir: Union[str, Path]) -> None:
        checkpoint = self.checkpoint_load_latest(output_dir)
        if checkpoint is None:
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
    
    def checkpoint_load_latest_and_resume_from_wandb(self, output_dir: Union[str, Path]) -> None:     
        import wandb
        api = wandb.Api()
        run = api.run(self.cfg.trainer.wandb_run_name)
        artifact = run.use_artifact(self.cfg.trainer.wandb_artifact_name)
        artifact_dir = artifact.download()
        checkpoint = self.checkpoint_load_latest(artifact_dir)
        if checkpoint is None:
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
    ###End Copilot generated 


class MultiLossTrainer(BaseMultiTrainer):
    ''' First usable trainer, for tabular data with multiple losses'''
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)


    def setup_extras(self) -> None:
        # eut.init_plot_options(self.cfg)
        eut.set_project_utils_options(self.cfg)

    def train(self) -> None:

        LOG.info(
            f"Starting on node {self.cfg.trainer.rank}, device {self.train_device}")
        starting_epoch = self.current_epoch
        time_start = time.time()
        for epoch in range(starting_epoch, self.cfg.trainer.epochs):
            self.current_epoch = epoch
            self.model_hook()  # Customizable hook defined in trainers, mainly for freezing-unfreezing
            step_dicts = []
            LOG.info(
                f"--------- {self.cfg.trainer.rank}:{self.train_device} - epoch: {epoch} --------- ")
            for iteration, input in enumerate(self.train_dataloader):
                step_dict = self.training_step(iteration, input) # train on batch
                step_dicts.append(step_dict)
                self.log_iteration(iteration, epoch, step_dict)


            self.log_epoch(epoch, iteration, step_dicts)
            epoch_logs = self.val(epoch, iteration)
            if epoch % self.cfg.trainer.log.check_every_n_epoch == 0:
                # This operation takes a fews seconds (overhead)
                self.checkpoint_dump(epoch=epoch)

        self.checkpoint_dump(epoch=self.current_epoch)
        self.register_model(epoch_logs)

        # Log time to run
        time_end = time.time() - time_start
        time_print = time.strftime("%H:%M:%S", time.gmtime(time_end))
        LOG.info(
            f"{self.cfg.trainer.rank}:{self.train_device} training finished in {time_print}")
        if self.writer is not None:
            self.writer.add_scalar("Time_End", time_end/60,
                                    (self.current_epoch+1))

        return 0

    def training_step(self, iteration, input):
        self.optimizer.zero_grad()
        outputs, labels, size = self.infer_model(input)
        outputs, labels = self.model.preprocess_shapes(outputs, labels)
        losses = self.criterion(outputs, labels, self.model) # pass the model in case of l1 (Lasso) loss
        scores = self.model.scoring(outputs, labels) # measures scores such as accuracy
        total_loss = 0
        for loss in losses.values():
            total_loss += loss
        total_loss.backward()
        self.optimizer.step()
        losses.update({'total': total_loss})
        step_dict = {'size': size}
        step_dict.update({'losses': {}})
        step_dict.update({'scores': {}})
        for name, loss in losses.items():
            step_dict['losses'].update({f'{name}': loss.detach()})
        for name, score in scores.items():
            step_dict['scores'].update({f'{name}': score.detach()})
        return step_dict

    def infer_model(self, input):
        "Key Lines of Code managing input, labels and forward pass; facilitates class override"
        " Depending on Task and model, ouputs and labels must be a dict, whose elements must match the loss function"
        (X, Z, labels) = self.process_inputs(input)
        # outputs = self.model(X, Z) #Depricated
        outputs = self.model(X)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice': labels}, size

    def process_inputs(self, input):
        " Currently used for val, possibly add to infer model and override all trainers "
        (X, Z, labels) = input
        X = X.to(self.train_device)
        Z = Z.to(self.train_device)
        labels = labels.to(self.train_device)
        X = self.masker.mask_tabular(X) # Masking is done here for tabular data, not in dataloader because we often need original data
        return (X, Z, labels)

    def val(self, epoch, iteration):
        # Epoch and iteration are kept for logging at same point as Training logs
        self.model.eval()
        step_dicts = []
        for idx, input in enumerate(self.val_dataloader):
            step_dict = self.eval_step(input)
            step_dicts.append(step_dict)
        self.log_epoch(epoch, iteration, step_dicts, subset='Val')
        input = self.process_inputs(input)
        self.model.special_log(self.get_cloud_logger(), input, self.current_epoch)
        self.model.train()


### --------- Functions below are more Project specific --------------------------

    def get_cloud_logger(self):
        if self.cfg.trainer.use_clearml:
            cloud_logger='clearml'
        elif self.cfg.trainer.use_wandb:
            cloud_logger='wandb'
        else: cloud_logger=None
        return cloud_logger
    
    def register_model(self, epoch_logs):
        ''' Register model in a dictionary, and save to json '''
        ''' Useful for quick access to best models - tested but not verified consistently'''
        if 'debug' in self.cfg.trainer.name: return 0 # do not register debug models

        # model_info_path = os.path.join(get_original_cwd(), self.cfg.trainer.model_registry)
        model_info_path = to_absolute_path(self.cfg.trainer.model_registry)
        with open(model_info_path, 'r') as f:
            models = json.load(f)
        if self.cfg.trainer.use_wandb:
            import wandb # once imported and init elsewhere, can access the singleton instance
            wandb_id = wandb.run.id
        else: wandb_id = None

        model_path = self.cfg.trainer.checkpointpath_save
        last_model_info = {'epoch': self.current_epoch, 'logs': epoch_logs, 'model_path': model_path, 'wandb_id': wandb_id}
        # last_model_info = {'epoch': self.current_epoch, 'logs': epoch_logs, 'model_path': os.path.abspath(self.cfg.trainer.checkpointpath), 'wandb_id': wandb_id}
        models.update({self.cfg.trainer.name: {'last': last_model_info}})
        # Check if update best model in registry dict:
        if models[self.cfg.trainer.name].get('best') is not None:
            # TODO: add a function in utils to select best comparing metric for each model
            if models[self.cfg.trainer.name]['best']['logs']['Loss/Val_total'] > epoch_logs['Loss/Val_total']:
                models[self.cfg.trainer.name].update({'best': last_model_info})
        else:
            models[self.cfg.trainer.name].update({'best': last_model_info})

        with open(model_info_path, 'w') as f:
            json.dump(models, f)
        

    def eval(self):
        ''' Loop over test dataset for metrics - also gets hessian and logs necessary info'''
        # Loop on Test data to get Hessian if available:
        self.model.freeze(False) # confirm all grads required for hessian backprop
        for name, param in self.model.named_parameters():
            if ('utility' in name) and ('image' not in name):
                stds, exclude = eut.get_utility_Hessian(
                    self.cfg, self, layer_name=name)
                cloud_logger = self.get_cloud_logger()
                eut.log_Betas(self.model, self.cfg.data.X_cols,
                                stds, name, cloud_logger=cloud_logger)

        # Get Test data statisitics:
        step_dicts = []
        for idx, input in enumerate(self.test_dataloader):
            step_dict = self.eval_step(input)
            if self.cfg.trainer.eval.beta_r_stats:
                beta_outputs_dict = eut.get_beta_contributions(
                    self.cfg.data.X_cols, input, self.train_device, self)

                r_beta_metrics_dict = eut.get_r_beta_metrics(
                    step_dict['representations'], beta_outputs_dict, input, self.cfg.data.X_cols,
                    self.cfg.trainer.masked_tabular_list, self.train_device)
                step_dict.update({'r_beta': r_beta_metrics_dict})
            step_dicts.append(step_dict)

        self.log_epoch(self.current_epoch, len(
            self.test_dataloader), step_dicts, subset='Test')

        if self.cfg.trainer.eval.beta_r_stats:
            if self.cfg.trainer.use_clearml:
                cloud_logger='clearml'
            elif self.cfg.trainer.use_wandb:
                cloud_logger='wandb'
            else: cloud_logger=None
            eut.log_r_beta_metrics(step_dicts, 0, cloud_logger)
        
        return 0

    def eval_step(self, input):
        ''' Evaluate Scores and losses, but also returns values from the image model that enter the utility'''
        # Get values of  representation outputs during forward pass (Beta Vs R experiments)
        hooked_layer = self.model.return_r_layer()
        if hooked_layer is not None:
            hooked_outputs = {}

            def getOutputs_hook(name):
              # the hook signature
              def hook(model, input, output):
                hooked_outputs[name] = output.detach()
              return hook
            h = hooked_layer.register_forward_hook(
                getOutputs_hook('representations'))
        # Model inference
        with torch.no_grad():
            outputs, labels, size = self.infer_model(input)
            outputs, labels = self.model.preprocess_shapes(outputs, labels)
            losses = self.criterion(outputs, labels, self.model)
            scores = self.model.scoring(outputs, labels)
            total_loss = 0
            for loss in losses.values():
                total_loss += loss
            losses.update({'total': total_loss})
            step_dict = {'size': size}
            step_dict.update({'losses': {}})
            step_dict.update({'scores': {}})
            for name, loss in losses.items():
                step_dict['losses'].update({f'{name}': loss.detach()})
            for name, score in scores.items():
                step_dict['scores'].update({f'{name}': score.detach()})
        if hooked_layer is not None:
            step_dict.update(
                {'representations': hooked_outputs['representations']})
        return step_dict

    def hessian_step(self, input):
        # The hessian needs steps with loss attached to grads
        # self.model.to(self.train_device)
        outputs, labels, size = self.infer_model(input)
        outputs, labels = self.model.preprocess_shapes(outputs, labels)
        losses = self.criterion(outputs, labels, self.model)
        step_dict = {
            'loss': losses['cce'], 'size': size}
        return step_dict

    def model_hook(self):
        ''' Perform model specific modifications - beginning of each epoch '''
        ''' Best coding would simply call model.hook '''
        if self.cfg.trainer.freeze_on_epoch is not None:
            if self.cfg.trainer.freeze_on_epoch == self.current_epoch:
                # Freezing method depends on the model
                self.model.freeze()
        if self.cfg.trainer.unfreeze_on_epoch is not None:
            if self.cfg.trainer.unfreeze_on_epoch == self.current_epoch:
                self.model.freeze(False)
        # Model specific code - Reversal models
        if self.cfg.trainer.encParam.reverse_epoch is not None:
            if self.cfg.trainer.encParam.reverse_epoch == self.current_epoch:
                # Only implemented for Reversal Models
                self.model.reverse(self.cfg.trainer.param.reverse_lbda_multiply)


    def debug_stop_training(self, iteration, input):
        # Emergency stop training and save states for debug - deprecated
        LOG.error('Nan encountered in loss term, exiting training')
        self.checkpoint_dump(epoch='NAN')
        debug_dict = {'iteration': iteration, 'input': input}
        file_path = os.path.join(self.cfg.trainer.output_dir, 'debug_dict.pkl')
        with open(file_path, 'wb') as f:
            pickle.dump(debug_dict, f)
        raise ValueError

    def log_iteration(self, iteration, epoch, metrics):
        ''' Add Beta Logging Support '''
        super(MultiLossTrainer, self).log_iteration(iteration, epoch, metrics)
        if iteration % self.cfg.trainer.log.every_n_iter == 0:
            self.log_betas(iteration, epoch)

    def log_betas(self, iteration, epoch):
        ''' Note: get_weight_statistics does the named_param loop too; define layer name in cfg to be more consistent?'''
        if self.writer is not None:
            for name, parm in self.model.named_parameters():
                if ('utility' in name) and ('image' not in name):
                    layer_name = name
                    weights_stats = eut.get_weight_statistics(
                            self.model, layer_name, self.cfg.data.X_cols, self.cfg.trainer.masked_tabular_list,
                            self.cfg.trainer.eval.normalizing_feature, self.train_device)
                    for key, value in weights_stats.items():
                        self.writer.add_scalar(key, value,
                                               epoch*len(self.train_dataloader)+iteration)


class ImTester_trainer(MultiLossTrainer):
    ''' Trainer for imtester model, uses only tabular inforamtion but simulates masking, redundancy etc... '''
    def infer_model(self, input):
        (X, Z, labels) = input
        X = X.to(self.train_device)
        Z = Z.to(self.train_device)
        labels = labels.to(self.train_device)
        X_true = X.clone().detach().requires_grad_(True)
        X = self.masker.mask_tabular(X) #? ????iNVESTIGATE ALL THIS ??? + Control output scales
        # outputs = self.model(X, Z) # Depricated
        outputs = self.model(X, X_true)
        X_masked = X_true - X
        non_empty_mask = X_masked.abs().sum(dim=0).bool()
        masked = X_masked[:, non_empty_mask] 
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice': labels, 'X': X.clone().detach(), 'masked': masked.detach()}, size
    

class LMNL_trainer(MultiLossTrainer):
    ''' L-MNL in the original sense (2020), no image model, and using a Z tabular variables independant of X '''
    def infer_model(self, input):
        "Key Lines of Code managing input, labels and forward pass; facilitates class override"
        " Depending on Task and model, ouputs and labels must be a list, whose elements must match the loss function"
        (X, Z, labels) = self.process_inputs(input)
        outputs = self.model(X, Z)
        size = torch.tensor(len(labels)).float().to(self.train_device)
        return outputs, {'choice': labels}, size