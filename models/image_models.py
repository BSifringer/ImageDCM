from omegaconf import open_dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_utils as mut
from torchvision import models
import torchvision
from torchvision.utils import make_grid
import logging
from abc import abstractmethod

LOG = logging.getLogger(__name__)

''' From https://github.com/tadeephuy/GradientReversal '''
from .gradient_reversal import GradientReversal

class Backbone_Model(nn.Module):
    ''' Core abstract class for image models - prepares the backbone module to output its feature space'''

    def __init__(self, cfg, num_labels):
        super(Backbone_Model, self).__init__()
        '''  Get and prepare Backbone to Num_labels: '''
        self.backbone, self.layer_name, self.lay_idx = mut.get_backbone(cfg)
        ''' Setup new output layer of backbone for classification '''
        if self.lay_idx is None:
            self.backbone._modules[self.layer_name] = nn.Identity()
        else:
            self.backbone._modules[self.layer_name][self.lay_idx] = nn.Identity(
            )
        try:
            x = torch.randn(1, 3, cfg.data.img_height,
                            cfg.data.img_width)
        except:
            LOG.error(
                'image dimensions not instantiated -\n Common error: forgot to Call image data configuration e.g., data=image_data')
            raise

        output = self.backbone(x)
        mut.set_parameter_requires_grad(
            self.backbone, requires_grad=cfg.trainer.train_backbone)
        self.num_features = output.shape[-1]
        LOG.debug('Backbone Model Final layer features: {}'.format(
            self.num_features))
            
        ''' Get saliency layer '''
        self.saliency_layer_name = mut.get_saliency_layer(cfg)

    @ abstractmethod
    def forward(self):
        pass

    def return_r_layer(self):
        if self.lay_idx is None:
            return self.backbone._modules[self.layer_name]
        else:
            return self.backbone._modules[self.layer_name][self.lay_idx]

    def return_saliency_layer(self):
        return self.backbone._modules[self.saliency_layer_name]

    # Currently redundant with mut.set_parameter_requires_grad; TODO : refactor if needed
    def freeze(self, freeze_layers=True):
        ''' Classes inhreiting from Backbone wil ONLY have the backbone frozen (!careful new named layers) '''
        # True will freeze, False will unfreeze
        requires_grad = not freeze_layers
        mut.set_parameter_requires_grad(
            self.backbone, requires_grad=requires_grad)

    @abstractmethod
    def special_log(*args):
        ''' Log elements specific to the model'''
        pass

class Detection_Model(Backbone_Model):
    ''' This Model Detects original ground-truths from generated image '''

    def __init__(self, cfg):
        self.fuse_detection_labels = cfg.trainer.fuse_image_detection_labels
        self.num_labels = cfg.trainer.param.n_betas * \
            (2-self.fuse_detection_labels)
        super(Detection_Model, self).__init__(cfg, self.num_labels)
        # New: split backbone and detection layer
        self.detection_layer = nn.Linear(int(self.num_features), int(self.num_labels))

        if cfg.trainer.encParam.activation == 'tanh':
            self.activation = nn.Tanh()
        elif cfg.trainer.encParam.activation == 'relu':
            self.activation = nn.ReLU()
        elif cfg.trainer.encParam.activation == 'identity':
            self.activation = nn.Identity()
        else: raise NotImplementedError

    def forward(self, images):
        features = self.activation(self.backbone(images))
        return {'detected': self.detection_layer(features)}

    def return_r_layer(self):
        ''' This model has no representation term '''
        return None

    def preprocess_shapes(self, outputs, labels):
        if self.fuse_detection_labels:
            labels['X'] = labels['X'].sum(dim=-2)
        labels['X'] = labels['X'].view(outputs['detected'].shape)
        return outputs, labels

    @staticmethod
    def scoring(outputs, labels):
        """ 1 attribute can have multiple counts. The accuracy is multiplied by the batch size for (score) correct averaging by main loop"""
        LOG.debug('Outputs and Labels example: {}\n {}\n {}'.format(
            outputs['detected'][0], torch.round(outputs['detected'][0]), labels['X'][0]))
        LOG.debug('Bool vector and sum {}, total: {} of {}'.format(torch.round(
            outputs['detected'][0]) == labels['X'][0], (torch.round(outputs['detected']) == labels['X']).sum(), len(labels['X'].view(-1))))
        detectables = torch.abs(labels['X']).sum()
        diff = torch.abs(torch.round(outputs['detected']) - labels['X']).sum()
        LOG.debug('Detectables {} vs Error {}'.format(detectables, diff))
        return {'detect': (detectables-diff) / detectables * len(labels['X'])}


class Image_Choice_Model(Backbone_Model):
    ''' This Model predicts choice based on Image only '''

    def __init__(self, cfg):
        # The choice is binary:
        self.num_labels = cfg.data.n_choice
        super(Image_Choice_Model, self).__init__(cfg, self.num_labels)
        if cfg.trainer.encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.num_features), int(self.num_labels), bias=False)
        elif cfg.trainer.encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.num_features), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, self.num_labels, bias=False))
        else: raise NotImplementedError
       
        #Activation from feature to latent space
        if cfg.trainer.encParam.activation == 'tanh':
            self.activation = nn.Tanh()
        elif cfg.trainer.encParam.activation == 'relu':
            self.activation = nn.ReLU()
        elif cfg.trainer.encParam.activation == 'identity':
            self.activation = nn.Identity()
        else: raise NotImplementedError

    def forward(self, images):
        features = self.activation(self.backbone(images))
        return {'choice' : self.choice_layer(features)}

    @staticmethod
    def preprocess_shapes(outputs, labels):
        ''' Turn one hot labels to class encoding, views for CCE loss, same as Tab models '''
        labels['choice'] = labels['choice'].view(-1, 2).argmax(-1)
        try:
            outputs['choice'] = outputs['choice'].view(-1, 2)
        except:
            import pdb; pdb.set_trace()
        return outputs, labels

    @staticmethod
    def scoring(outputs, labels):
        ''' Score is defined here as a count before it is divided by batch size for obtaining accuracy, same as Tab models '''
        return {'choice': (torch.max(outputs['choice'], 1)[1].view(
            labels['choice'].size()) == labels['choice']).sum()}


class Image_Extract_Model(Image_Choice_Model):
    ''' Added bottle neck'''
    def __init__(self, cfg):
        # We want the latent space to be as big as extra dimensions only:
        encParam = cfg.trainer.encParam
        self.num_labels = encParam.added_latent_dim
        self.added_latent_dim = encParam.added_latent_dim
        super(Image_Choice_Model, self).__init__(cfg, self.num_labels)


        ### Layer from latent with no detect - to choice
        if encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.added_latent_dim), 2, bias=False)
        elif encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.added_latent_dim), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, 2, bias=False))
        elif encParam.choice_layer == 'homogeneous':
            try:
                assert (self.added_latent_dim%2 == 0)
            except AssertionError:
                raise AssertionError('Latent dim must be even for homogeneous betas (because choice is binary)')
            self.choice_layer = Homogeneous_betas(int(self.added_latent_dim/2))
        else: raise NotImplementedError

        #Add bottleneck to backbone (new refactor)
        self.bottle_neck = nn.Linear(int(self.num_features), int(self.added_latent_dim))

    def forward(self, images):
        feature_space = self.backbone(images)
        z_mean = self.activation(self.bottle_neck(feature_space))
        choice = self.choice_layer(z_mean)
        return {'choice':choice}


    def return_r_layer(self):
        return self.choice_layer

    def freeze(self, freeze_layers=True):
        ''' Freezing choice layer or not changes the experiment, up to the designer '''
        requires_grad = not freeze_layers
        mut.set_parameter_requires_grad(self.backbone, requires_grad)
        mut.set_parameter_requires_grad(self.choice_layer, requires_grad)


class ImCorr_Model(nn.Module):
    ''' Image model and choice model are stacked with a simple addition for choice prediction'''
    def __init__(self, Image_cls, Tab_cls,  cfg):
        super(ImCorr_Model,  self).__init__()
        self.image_model = Image_cls(cfg)
        self.dcm = Tab_cls(cfg)
        self.freeze_dcm = cfg.trainer.freeze_dcm

    def forward(self, X, images):
        tab_outputs = self.dcm.forward(X)
        image_representation = self.image_model(images)
        return {'choice': image_representation['choice'] + tab_outputs['choice'].view(-1, 2), 'choice_dcm': tab_outputs['choice'].view(-1, 2)}

    def load_state_dict(self, state_dict):
        ''' Override to allow partial model checkpoint loading '''
        ''' Return True instead of None if custom load is used '''
        custom_load = None
        try:
            super(ImCorr_Model, self).load_state_dict(state_dict)
        except:
            try:
                self.dcm.load_state_dict(state_dict)
                LOG.warning(' Loaded only DCM weights ')
                custom_load = True
            except:
                try:
                    ''' This method is safer than loading with strict=False, which could load nothing.
                    This will work if only a single layer is the same, which may not be desired - alwyas double check the warning'''
                    #From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
                    model_dict = self.image_model.state_dict()
                    # 1. filter out unnecessary keys
                    state_dict = {
                        k: v for k, v in state_dict.items() if k in model_dict}
                    if not state_dict:
                        LOG.error(
                            'Cannot use this Checkpoint - no matching Keys from state dict')
                        raise ValueError
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(state_dict)
                    # 3. load the new state dict
                    self.image_model.load_state_dict(model_dict)
                    LOG.warning(
                        'Loading Only Image Model weights - updated layers = {}'.format(len(state_dict)))
                    custom_load = True
                except:
                    LOG.error('Cannot use this State Dict')
                    raise
        return custom_load

    @staticmethod
    def preprocess_shapes(outputs, labels):
        ''' Turn one hot labels to class encoding, views for CCE loss, same as Tab models '''
        labels['choice'] = labels['choice'].view(-1, 2).argmax(-1)
        try:
            outputs['choice'] = outputs['choice'].view(-1, 2)
        except:
            import pdb; pdb.set_trace()
        return outputs, labels

    @staticmethod
    def scoring(outputs, labels):
        ''' Score is defined here as a count before it is divided by batch size for obtaining accuracy, same as Tab models '''
        return {'choice': (torch.max(outputs['choice'], 1)[1].view(
            labels['choice'].size()) == labels['choice']).sum()}

    def return_r_layer(self):
        return self.image_model.return_r_layer()

    def return_saliency_layer(self):
        return self.image_model.return_saliency_layer()

    def freeze(self, freeze_layers=True):
        # True will freeze, False will unfreeze
        requires_grad = not freeze_layers
        if not self.freeze_dcm:
            self.image_model.freeze(freeze_layers)
        else:
            mut.set_parameter_requires_grad(
                self.dcm, requires_grad=requires_grad)
            
    def special_log(self, cloud_logger, input, epoch):
        # log a sample of the image input:
        X, Z, labels, images = input
        if cloud_logger == 'clearml':
            from clearml import Task
            task = Task.current_task()
            logger = task.get_logger()
            logger.report_image(
                'sample_images', 'sample_images', iteration=epoch, image=images[0])
        elif cloud_logger=='wandb':
            import wandb
            wandb.log({'sample_images': wandb.Image(images[0], caption='Sample Image')})
        else:
            LOG.warning('No cloud logger specified')
        return       

class MultiTargetModels(Detection_Model):
    ''' Abstract Class for Models using Both Choice and Detection '''

    def __init__(self, cfg):
        super(MultiTargetModels, self).__init__(cfg)

    def preprocess_shapes(self, outputs, labels):
        ''' This function takes into account self.forward output and trainer's inference function (e.g. Choice_Detection_trainer) '''
        if self.fuse_detection_labels:
            labels['X'] = labels['X'].sum(dim=-2)
        outputs['detected'] = outputs['detected'].view(-1, self.num_labels)
        labels['X'] = labels['X'].view(outputs['detected'].shape)
        labels['choice'] = labels['choice'].view(-1, 2).argmax(-1)
        outputs['choice'] = outputs['choice'].view(-1, 2)
        return outputs, labels


    @ staticmethod
    def scoring(outputs, labels):
        """ 1 attribute can have multiple counts. The accuracy is multiplied by the batch size for (score) correct averaging by main loop"""
        img_utility, detected  = outputs['choice'].detach(), outputs['detected'].detach()
        label_choice, X = labels['choice'], labels['X']
        detectables = torch.abs(X).sum()
        diff = torch.abs(torch.round(detected) - X).sum()
        return {'detect': (detectables-diff) / detectables * len(X),
                'choice': (torch.max(img_utility, 1)[1].view(label_choice.size()) == label_choice).sum()}


class Detection2Choice_Model(MultiTargetModels):
    ''' Detect Pedestrians and turn to utility without use of tabular as input
    Use case: trainer.type=choice_detect     trainer.im_model=detect2choice
    '''

    def __init__(self, cfg):
        super(Detection2Choice_Model, self).__init__(cfg)
        self.n_betas = cfg.trainer.param.n_betas
        self.img_utility = nn.Linear(
            self.n_betas, 1, bias=False)
        try:
            assert self.fuse_detection_labels == False
        except AssertionError:
            LOG.error(
                'Detection Labels in this Model cannot be fused => Utility for binary choice is built upon prediction')
            raise 

    def forward(self, images):
        detected_objects = self.detection_layer(self.activation(self.backbone(images))).view(-1, 2, self.n_betas)
        img_utility = self.img_utility(detected_objects)
        return {'choice': img_utility, 'detected': detected_objects}

    def return_r_layer(self):
        return self.img_utility

    def freeze(self, freeze_layers=True):
        ''' Freezing Image utility  or not changes the experiment, up to the designer '''
        requires_grad = not freeze_layers
        mut.set_parameter_requires_grad(self.backbone, requires_grad)
        mut.set_parameter_requires_grad(self.img_utility, requires_grad)


class ChoiceWithDetection_Model(MultiTargetModels):
    ''' Image Choice model while detecting as a subtask:
    Use case: trainer.type=choice_detect     trainer.im_model=choice_w_detect
    '''

    def __init__(self, cfg):
        super(ChoiceWithDetection_Model, self).__init__(cfg)
        if cfg.trainer.encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.num_features), int(cfg.data.n_choice), bias=False)
        elif cfg.trainer.encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.num_features), int(cfg.data.n_choice)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(cfg.data.n_choice)*cfg.trainer.encParam.seq_scale, cfg.data.n_choice, bias=False))
        else: raise NotImplementedError
    
        self.unfreeze_choice_layer = cfg.trainer.encParam.unfreeze_choice_layer

    def forward(self, images):
        feature_space = self.activation(self.backbone(images))
        detected_objects = self.detection_layer(feature_space)
        choice = self.choice_layer(feature_space)
        return {'choice': choice, 'detected': detected_objects}

    def return_r_layer(self):
        return self.choice_layer

    def freeze(self, freeze_layers=True):
        ''' Freezing choice layer or not changes the experiment, up to the designer '''
        requires_grad = not freeze_layers
        mut.set_parameter_requires_grad(self.backbone, requires_grad)
        mut.set_parameter_requires_grad(self.detection_layer, requires_grad)
        if not self.unfreeze_choice_layer:
            mut.set_parameter_requires_grad(self.choice_layer, requires_grad)




class ImCorrDetect_Model(nn.Module):
    ''' Image model with detection subtask + tabular model'''
    def __init__(self, MultiImg_cls, Tab_cls, cfg):
        super(ImCorrDetect_Model,  self).__init__()
        self.image_model = MultiImg_cls(cfg)
        self.dcm = Tab_cls(cfg)
        self.freeze_dcm = cfg.trainer.freeze_dcm
        self.beta_loss = False
        if 'beta' in cfg.trainer.loss:
            self.beta_loss = (not cfg.trainer.fuse_image_detection_labels)
            if self.beta_loss:
                for name, layer in self.image_model.named_modules():
                    if 'detect2Choice' in name:
                        self.img_beta_layer = layer
            else:
                LOG.warning('Beta loss cannot be used with fused labels')
        

    def forward(self, X, images):
            outputs = self.image_model.forward(images)
            tab_outputs = self.dcm.forward(X)
            outputs['choice'] = outputs['choice']+tab_outputs['choice'].view(outputs['choice'].shape)
            if self.beta_loss:
                dcm_utility_weights = self.dcm.utility.weight
                image_utility_weights = self.img_beta_layer.weight
                weight_loss = torch.norm(dcm_utility_weights.detach()-image_utility_weights)
                outputs['beta'] = weight_loss
            return outputs
    
    def return_r_layer(self):
        return self.image_model.return_r_layer()

    def return_saliency_layer(self):
        return self.image_model.return_saliency_layer()

    ''' This model inherits Multi-Target models, thus preprocess shape and scoring work correctly '''

    def preprocess_shapes(self, outputs, labels):
        return self.image_model.preprocess_shapes(outputs, labels)

    def scoring(self, outputs, labels):
        return self.image_model.scoring(outputs, labels)

    def freeze(self, freeze_layers=True):
        ''' Freeze only the image model - by experiment design '''
        if not self.freeze_dcm:
            self.image_model.freeze(freeze_layers)
        else:
            requires_grad = not freeze_layers
            mut.set_parameter_requires_grad(self.dcm, requires_grad)

    def reverse(self, reverse_lbda):
        self.image_model.reverse(reverse_lbda)

    def load_state_dict(self, state_dict):
        ''' Override to allow partial model checkpoint loading '''
        ''' Return True instead of None if custom load is used '''
        custom_load = None
        try:
            super(ImCorrDetect_Model, self).load_state_dict(state_dict)
        except:
            try:
                self.dcm.load_state_dict(state_dict)
                LOG.warning(' Loaded only DCM weights ')
                custom_load = True
            except:
                try:
                    self.image_model.load_state_dict(state_dict)
                    LOG.warning('Loading Only Image Model weights - All layers successful')
                    custom_load = True
                except:
                    try:
                        ''' This method is safer than loading with strict=False, which could load nothing.
                        This will work if only a single layer is the same, which may not be desired - alwyas double check the warning'''
                        #From https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113
                        model_dict = self.image_model.state_dict()
                        # 1. filter out unnecessary keys
                        state_dict = {
                            k: v for k, v in state_dict.items() if k in model_dict}
                        if not state_dict:
                            LOG.error(
                                'Cannot use this Checkpoint - no matching Keys from state dict')
                            raise ValueError
                        # 2. overwrite entries in the existing state dict
                        model_dict.update(state_dict)
                        # 3. load the new state dict
                        self.image_model.load_state_dict(model_dict)
                        LOG.warning(
                            'Loading Only Image Model weights - updated layers = {}'.format(len(state_dict)))
                        custom_load = True
                    except:
                        LOG.error('Cannot use this State Dict')
                        raise
        return custom_load


    def special_log(self, cloud_logger, input, epoch):
        # log a sample of the image input:
        X, Z, labels, images = input
        if cloud_logger == 'clearml':
            from clearml import Task
            task = Task.current_task()
            logger = task.get_logger()
            logger.report_image(
                'sample_images', 'sample_images', iteration=epoch, image=images[0])
        elif cloud_logger=='wandb':
            import wandb
            wandb.log({'sample_images': wandb.Image(images[0], caption='Sample Image')})
        else:
            LOG.warning('No cloud logger specified')
        return       


class Latent_Sampler(nn.Module):
    def forward(self, z_mean, z_log_var, n_draws=20):
        batch, dim = z_mean.shape
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = torch.randn((int(batch), int(dim), int(n_draws))).to(z_mean.device)
        return z_mean + torch.mean(torch.exp(0.5 * z_log_var).unsqueeze(-1) * epsilon, dim=-1)

class Return_First(nn.Module):
    ''' Create a Module that returns the first argument (instead of doing any operation)'''
    def forward(self, first_arg, *args):
        return first_arg

class Return_Constant(nn.Module):
    def __init__(self, const):
        super(Return_Constant, self).__init__()
        self.const = const
    def forward(self, X):
        return (torch.ones(X.size())*self.const).to(X.device)

class Add(nn.Module):
    def forward(self, X, Y):
        return X+Y

class Homogeneous_betas(nn.Module):
    ''' Applies a same single linear layer to all dimensions of X'''
    def __init__(self, num_features):
        super(Homogeneous_betas, self).__init__()
        self.num_features = num_features
        self.image_utility_layer = nn.Linear(num_features, 1, bias=False)

    def forward(self, X):
        shape = X.shape
        list_shape = [i for i in shape]
        list_shape[-1] = -1
        list_shape.append(self.num_features)
        choice = self.image_utility_layer(X.view(list_shape))
        return choice.view(list_shape[:-1]) 



class Encoder_Model(MultiTargetModels):
    ''' This model serves to create latent representations of the Image data with z_mean, z_sigma '''
    def __init__(self, cfg):
        # We want the latent space to be as big as all independent features + extra dimensions for added information:
        super(Encoder_Model, self).__init__(cfg)
        if self.lay_idx is None:
            self.backbone._modules[self.layer_name] = nn.Identity()
        else:
            self.backbone._modules[self.layer_name][self.lay_idx] = nn.Identity(
            )
        self.added_latent_dim = cfg.trainer.encParam.added_latent_dim * cfg.trainer.encParam.latent_dim_multiplier
        self.num_latent_space = self.num_labels * cfg.trainer.encParam.latent_dim_multiplier + self.added_latent_dim

        self.z_mean_layer = nn.Linear(int(self.num_features), int(self.num_latent_space))
        self.z_log_sigma_sq_layer = nn.Linear(int(self.num_features), int(self.num_latent_space))


        encParam = cfg.trainer.encParam

        #Activation from feature to latent space
        if encParam.activation == 'tanh':
            self.activation = nn.Tanh()
        elif encParam.activation == 'relu':
            self.activation = nn.ReLU()
        elif encParam.activation == 'identity':
            self.activation = nn.Identity()
        else: raise NotImplementedError

        ### Layer from latent 2 detect
        if encParam.detect_layer == 'linear':
            self.detection_layer = nn.Linear(int(self.num_latent_space-self.added_latent_dim), int(self.num_labels))
        elif encParam.detect_layer == 'sequential':
            self.detection_layer = nn.Sequential(
                    nn.Linear(int(self.num_latent_space-self.added_latent_dim), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, int(self.num_labels)))
        elif encParam.detect_layer == 'identity':
            self.detection_layer = nn.Identity()
        else: raise NotImplementedError

        ### Layer from latent with no detect - to choice
        if encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.added_latent_dim), 2, bias=False)
        elif encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.added_latent_dim), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, 2, bias=False))
        else: raise NotImplementedError

        self.log_sigma_sq_scaler = cfg.trainer.encParam.log_sigma_sq_scaler
        if cfg.trainer.fuse_image_detection_labels:
            self.detect2Choice = nn.Linear(int(self.num_labels), 2, bias=False)
        else:
            self.detect2Choice = Homogeneous_betas(cfg.trainer.param.n_betas)
        ### Sampler for distributions z_mean, z_sig
        if encParam.sampling==True:
            self.sampling = Latent_Sampler()
            self.process_sigma = nn.Identity()
        else:
            self.sampling = Return_First()
            self.process_sigma = Return_Constant(self.log_sigma_sq_scaler)

        if encParam.detect2Choice==True:
            self.connect_layer = Add()
        else: self.connect_layer = Return_First()


    def forward(self, images):
        feature_space = self.backbone(images)
        z_mean = self.activation(self.z_mean_layer(feature_space))

        z_log_sigma_sq = self.z_log_sigma_sq_layer(feature_space)
        z_log_sigma_sq = self.process_sigma(z_log_sigma_sq)

        sample = self.sampling(z_mean, z_log_sigma_sq)
        correlated, invariant = sample[:,:-self.added_latent_dim], sample[:,-self.added_latent_dim:]

        detected_objects = self.detection_layer(correlated)
        choice = self.choice_layer(invariant)
        detect2choice = self.detect2Choice(detected_objects)
        final_choice = self.connect_layer(choice, detect2choice)
        return {'choice':final_choice, 'detected': detected_objects, 'Z':[z_mean, z_log_sigma_sq]}


    def return_r_layer(self):
        return self.choice_layer

    def freeze(self, freeze_layers=True):
        ''' Freezing choice layer or not changes the experiment, up to the designer '''
        requires_grad = not freeze_layers
        mut.set_parameter_requires_grad(self.backbone, requires_grad)
        mut.set_parameter_requires_grad(self.z_mean_layer, requires_grad)
        mut.set_parameter_requires_grad(self.z_log_sigma_sq_layer, requires_grad)
        mut.set_parameter_requires_grad(self.choice_layer, requires_grad)
        mut.set_parameter_requires_grad(self.detection_layer, requires_grad)
        mut.set_parameter_requires_grad(self.detect2Choice, requires_grad)



class Encoder_Reversal(Encoder_Model):
    ''' No KL loss, but reversal gradient instead. Output detect2choice for a seperate cce loss'''
    def __init__(self, cfg):

        super(Encoder_Reversal, self).__init__(cfg)
        if cfg.trainer.encParam.reversal==True:
            self.reversal_layer = GradientReversal(alpha=1.)
        else:
            self.reversal_layer = nn.Identity()

        encParam = cfg.trainer.encParam

        # Make detect and choice layer receive same input size:
        if encParam.detect_layer == 'linear':
            self.detection_layer = nn.Linear(int(self.num_latent_space), int(self.num_labels))
        elif encParam.detect_layer == 'sequential':
            self.detection_layer = nn.Sequential(
                    nn.Linear(int(self.num_latent_space), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, int(self.num_labels)))
        elif encParam.detect_layer == 'identity':
            self.detection_layer = nn.Identity()
        else: raise NotImplementedError

        ### Layer from latent with no detect - to choice
        if encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.num_latent_space), 2, bias=False)
        elif encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.num_latent_space), int(self.num_labels)*cfg.trainer.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.num_labels)*cfg.trainer.encParam.seq_scale, 2, bias=False))
        else: raise NotImplementedError

    def forward(self, images):
        feature_space = self.backbone(images)
        z_mean = self.activation(self.z_mean_layer(feature_space))

        z_log_sigma_sq = self.z_log_sigma_sq_layer(feature_space)
        z_log_sigma_sq = self.process_sigma(z_log_sigma_sq)

        sample = self.sampling(z_mean, z_log_sigma_sq) # samnples or returns mean only

        detection_head = self.reversal_layer(sample)
        detected_objects = self.detection_layer(detection_head)
        choice = self.choice_layer(sample)
        detect2choice = self.detect2Choice(detected_objects)
        final_choice = self.connect_layer(choice, detect2choice) # choice(default) or  choice+detect2choice
        return {'choice': final_choice, 'detected': detected_objects, 'latent': detect2choice}

    def reverse(self, reverse_lbda):
        self.reversal_layer = GradientReversal(alpha=reverse_lbda)


class Multiply_by_scalar(nn.Module):
    def __init__(self,):
        super(Multiply_by_scalar, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(1.))

    def forward(self, X):
        return X*self.scalar

class Latent_Choice_Model(Encoder_Model):
    ''' This Image model detects tabular inputs and then use values to estimate a second choice loss '''
    def __init__(self, cfg):
        if cfg.trainer.fuse_image_detection_labels:
            LOG.warning('Latent_Choice_Model is designed to work with fuse_detection_labels=False. This is not the case.')
            with open_dict(cfg.trainer):
                cfg.trainer.fuse_image_detection_labels = False

        super(Latent_Choice_Model, self).__init__(cfg)


        self.detect2Choice = Homogeneous_betas(cfg.trainer.param.n_betas)

        if cfg.trainer.encParam.detect2Choice==0:
            self.utility_scale_layer = Multiply_by_scalar()
        else:
            self.utility_scale_layer = nn.Identity()

        self.latent_choice_estimation_connection = Add()



    def forward(self, images):
        feature_space = self.backbone(images)
        z_mean = self.activation(self.z_mean_layer(feature_space))

        z_log_sigma_sq = self.z_log_sigma_sq_layer(feature_space)
        z_log_sigma_sq = self.process_sigma(z_log_sigma_sq)

        sample = self.sampling(z_mean, z_log_sigma_sq) # samnples or returns mean only
        correlated, invariant = sample[:,:-self.added_latent_dim], sample[:,-self.added_latent_dim:]

        detected_objects = self.detection_layer(correlated)
        choice = self.choice_layer(invariant)
        detect2choice = self.detect2Choice(detected_objects)

        final_choice = self.connect_layer(choice, detect2choice) # When Running Latent Corr with tabular, connectlayer returns choice only 
        final_choice = self.utility_scale_layer(nn.Tanh()(final_choice))

        latent_estimation = self.latent_choice_estimation_connection(choice, detect2choice)

       # Technically the latents are the detected object, and 'latent' is simply a second choice loss. But we keep our naming convention for simplicity 
        return {'choice':final_choice, 'detected': detected_objects, 'latent': latent_estimation}


class Contrastive_Model(Encoder_Reversal):
    ''' Testing models and code for contrastive similarity or distance between latent space of multi-modal inputs'''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.param = cfg.trainer.param
        self.n_layers = self.param.n_layers
        self.n_neurons = self.param.n_neurons
        self.dropout_prob = self.param.dropout_prob
        self.first = nn.Linear(self.param.n_betas, self.n_neurons) 
        self.linears = nn.Sequential()
        for i in range(self.n_layers-1):
            self.linears.add_module('linear{}'.format(
                i+1), nn.Linear(self.n_neurons, self.n_neurons))
            self.linears.add_module("relu{}".format(i+1), nn.ReLU())
            self.linears.add_module("dropout{}".format(
                i+1), nn.Dropout(self.dropout_prob))
        self.latent_space = nn.Linear(self.n_neurons, self.num_latent_space//2, bias=False)
        self.last = nn.Linear(self.param.n_betas, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

        # self.tab_encoder = nn.Sequential()


    def forward(self, X, images): # X first allows the us to use choice_detect_corr as a trainer for the contrastive model
        feature_space = self.backbone(images)
        z_mean = self.activation(self.z_mean_layer(feature_space))

        z_log_sigma_sq = self.z_log_sigma_sq_layer(feature_space)
        z_log_sigma_sq = self.process_sigma(z_log_sigma_sq)

        sample = self.sampling(z_mean, z_log_sigma_sq)
        
        r = F.relu(self.first(X))
        r = self.dropout(r)
        r = self.linears(r)
        tab_latent = self.activation(self.latent_space(r))
        product = (tab_latent.view(sample.shape)*sample).sum(dim=-1)
        difference = (tab_latent.view(sample.shape)-sample)
        product_diff = (tab_latent.view(sample.shape)*difference).sum(dim=-1)
        # contrast = - product.sum()/torch.norm(product+0.0001) + torch.abs(torch.norm(tab_latent)-torch.norm(sample))
        contrast = - product.sum()/torch.norm(product+0.0001) + product_diff.sum()/torch.norm(product_diff+0.0001)
        # contrast = torch.abs(torch.norm(tab_latent)-torch.norm(sample)) # This gets full choice accuracy (try same regularizer in choice_corr models)

        detected_objects = self.detection_layer(sample)
        choice = self.choice_layer(sample)
        # detect2choice = self.detect2Choice(detected_objects)

        tab_choice = self.choice_layer(tab_latent.view(sample.shape)) # Shared Decoder
        added_choice = self.choice_layer(difference)
        # final_choice = self.connect_layer(choice, tab_choice)
        final_choice = self.connect_layer(added_choice, tab_choice+choice)

        return {'choice': final_choice, 'detected': detected_objects, 'contrast': contrast }


class ImCorrContrastive_Model(ImCorrDetect_Model):
    ''' The ImCorrConstrastive_Model  is a ImCorrDetect_Model but receives an Constrastive in the init upon construction.
    This requires a change in the forward function for the adding X in the image model's forward function'''
    def forward(self, X, images):
            outputs = self.image_model.forward(X, images)
            tab_outputs = self.dcm.forward(X)
            outputs['choice'] = outputs['choice']+tab_outputs['choice'].view(outputs['choice'].shape)
            if self.beta_loss:
                dcm_utility_weights = self.dcm.utility.weight
                image_utility_weights = self.img_beta_layer.weight
                weight_loss = torch.norm(dcm_utility_weights.detach()-image_utility_weights)
                outputs['beta'] = weight_loss
            return outputs

