import torch
import numpy as np
import pandas as pd
import logging

from trainers import image_trainer, mixed_trainer


LOG = logging.getLogger(__name__)


def fix_random_seeds(seed: int = 42):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_trainer(trainer_type, cfg):
    ''' Based on Trainer type of experiment, chooses or creates correct class '''
    if trainer_type == 'survey':
        # trainer = survey_trainer.SurveyTrainer(cfg) # SurveyTrainer became MultiLossTrainer (can have single loss too) - Code refactored Project v.1
        trainer = mixed_trainer.MultiLossTrainer(cfg)
    elif trainer_type == 'lmnl':
        trainer = mixed_trainer.LMNL_trainer(cfg)
    elif trainer_type == 'imtester':
        trainer = mixed_trainer.ImTester_trainer(cfg)
    elif trainer_type == 'detection':
        trainer = image_trainer.Detection_trainer(cfg)
    elif trainer_type == 'gain':
        trainer = image_trainer.GAIN_trainer(cfg)
    elif trainer_type == 'img_choice':
        trainer = image_trainer.ImChoice_trainer(cfg)
    elif trainer_type == 'img_contrast':
        trainer = image_trainer.Contrastive_trainer(cfg)
    elif trainer_type == 'imCorr':
        trainer = image_trainer.ImCorr_trainer(cfg)
    elif trainer_type in ['choice_detect']:
        trainer = image_trainer.Choice_Detection_trainer(cfg)
    elif trainer_type in ['choice_detect_corr']:
        trainer = image_trainer.Choice_Detection_Corr_trainer(cfg)
    else:
        LOG.error(
            'Trainer type serves to chose which trainer is needed. Select a pre-existing one or add a line for {}'.format(trainer_type))
        raise NotImplementedError
    return trainer


''' Getting the KL loss as a regularizer (from conditional paper - this is not B-VAE) '''
'''https://github.com/dcmoyer/invariance-tutorial/blob/master/tutorial.ipynb'''
def all_pairs_gaussian_kl(mu, sigma, add_third_term=False):
    sigma_sq = torch.square(sigma) + 1e-8
    sigma_sq_inv = torch.reciprocal(sigma_sq)
    first_term = torch.matmul(sigma_sq,(sigma_sq_inv.t()))

    r = torch.matmul(mu * mu, sigma_sq_inv.t())
    r2 = mu * mu * sigma_sq_inv
    r2 = torch.sum(r2,1)

    second_term = 2*torch.matmul(mu, (mu*sigma_sq_inv).t())
    second_term = r - second_term + r2.t()


    if(add_third_term):
        r = torch.sum(torch.log(sigma_sq),1)
        r = torch.reshape(r,[-1,1])
        third_term = r - r.t()
    else:
        third_term = 0

    return 0.5 * ( first_term + second_term + third_term )

def kl_conditional_and_marg(z_mean, z_log_sigma_sq, dim_z):
    z_sigma = torch.exp( 0.5 * z_log_sigma_sq )
    all_pairs_GKL = all_pairs_gaussian_kl(z_mean, z_sigma, True) - 0.5*dim_z
    return torch.mean(all_pairs_GKL)

def get_loss(loss_types, lambda_dict):
    ''' Based on config loss name, returns a function definition which returns dict of corresponding losses'''
    ''' First define loss functions, then create dict of loss in all_losses'''

    def bceLoss(outputs, labels):
        return {'bce': lambda_dict['bce']*torch.nn.BCEWithLogitsLoss()(outputs, labels)}

    def cceLoss(outputs, labels, key='cce'):
        return {key: lambda_dict[key]*torch.nn.CrossEntropyLoss()(outputs, labels)}

    def l2Loss(outputs, labels, key='l2'):
        ''' L2 loss for detection (regression) '''
        return {key: lambda_dict[key]*torch.nn.MSELoss()(outputs, labels)}
    
    def sigmoidLoss(outputs, labels, key='sigmoid'):
        ''' Sigmoid loss for detection (regression) '''
        # !! CAREFUL RED LIGHT NOT SUPPORTED YET    
        return {key: lambda_dict[key]*torch.nn.BCEWithLogitsLoss()(outputs, (labels>0).to(labels.dtype))}

    def l1Loss(model):
        ''' Apply loss on all parameters except DCM or layers named utility'''
        l1 = sum(p.abs().sum() for name,
                        p in model.named_parameters() if 'utility' not in name)
        return {'l1': lambda_dict['l1']*l1}

    def prior_loss(z_mean, z_log_sigma_sq):
        ''' This is KL BVAE latent prior loss  '''
        prior_loss = 1 + z_log_sigma_sq - torch.square(z_mean) - torch.exp(z_log_sigma_sq)
        prior_loss = torch.sum(prior_loss, dim=-1)
        prior_loss = torch.mean(prior_loss) # loss must be single valued -> tutorial  has a mistake? Tensorflow differs?
        prior_loss *= -0.5
  
        return  {'prior': lambda_dict['prior']*prior_loss}



    def KL_loss(z_mean, z_log_sigma_sq):
        kl_qzx_qz_loss = kl_conditional_and_marg(z_mean, z_log_sigma_sq, z_mean.shape[1])
        return {'KL': lambda_dict['KL']*kl_qzx_qz_loss}


    def contrastLoss(outputs):
        return {'contrast': lambda_dict['contrast']*torch.sum(outputs)}

    ''' Construct and Return Loss functions compatible as Criterion format '''
    def all_losses(outputs, labels, model=None):
        ''' each loss adds an entry in the loss_dict, modeling requires fetching correct keys from outputs and labels'''
        loss_dict = {}
        if 'cce' in loss_types: # Main Loss
            loss_dict.update(cceLoss(outputs['choice'], labels['choice']))
        if 'cce_dcm' in loss_types: # Main Loss
            loss_dict.update(cceLoss(outputs['choice_dcm'], labels['choice'], key='cce_dcm'))
        if 'l2' in loss_types: # Detection Loss
            loss_dict.update(l2Loss(outputs['detected'], labels['X']))
        if 'l1' in loss_types: # L1 Regularization
            loss_dict.update(l1Loss(model))
        if 'prior' in loss_types: # Prior Loss of B-VAE
            loss_dict.update(prior_loss(*outputs['Z']))
        if 'KL' in loss_types: # KL Loss of adverserial paper
            loss_dict.update(KL_loss(*outputs['Z']))
        if 'contrast' in loss_types: # Loss for mulitplied vectors in feature space 
            loss_dict.update(contrastLoss(outputs['contrast']))
        if 'binary' in loss_types: # Deprecated
            loss_dict.update(bceLoss(outputs['choice'], labels['choice']))
        if 'cce2' in loss_types: # Choice loss for Image model (e.g., latent models)
            loss_dict.update(cceLoss(outputs['latent'], labels['choice'], key='cce2'))
        if 'beta' in loss_types: # Beta loss for two beta vectors
            loss_dict.update({'beta': lambda_dict['beta']*torch.nn.L1Loss()(outputs['beta'], torch.zeros_like(outputs['beta']))})
        if 'masked' in loss_types: # Test Loss for detecting masked inputs
            loss_dict.update(l2Loss(outputs['representation'], labels['masked'], key='masked'))
        if 'detect_masked' in loss_types: # Test Loss for detecting masked inputs
            loss_dict.update(l2Loss(outputs['detected_masked'], labels['X'], key='detect_masked'))
        if 'sigmoid' in loss_types: # Sigmoid loss for detection (multiclass)
            loss_dict.update(sigmoidLoss(outputs['detected'], labels['X'], key='sigmoid'))
        if loss_dict == {}:
            raise NotImplementedError('Loss type not implemented')
        return loss_dict
    
    return all_losses


def create_lambda_dict(params):
    ''' When adding a new loss, also add a new lambda to this dict'''
    return {
    'cce': params.cce_lbda,
    'cce_dcm': params.cce_dcm_lbda,
    'l2': params.l2_lbda,
    'bce': params.bce_lbda,
    'l1': params.l1_lbda,
    'prior': params.prior_lbda,
    'KL': params.KL_lbda,
    'cce2': params.cce2_lbda,
    'contrast': params.contrast_lbda,
    'beta': params.beta_lbda,
    'masked': params.masked_lbda,
    'detect_masked': params.detected_masked_lbda,
    'sigmoid': params.sigmoid_lbda
    }

