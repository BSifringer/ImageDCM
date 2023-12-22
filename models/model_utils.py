from models import tabular_models as tm
from models import image_models as im
import logging
import torchvision.models as vis

LOG = logging.getLogger(__name__)


''' Whenever a new model is defined, new rules to invoke it must be added here '''

def get_class(cls_name):
    ''' Returns non-initialized classes for composed models'''
    if cls_name == 'mnl':
        return tm.MNL
    if cls_name == 'choice_w_detection':
        return im.ChoiceWithDetection_Model
    if cls_name == 'dectection2choice':
        return im.Detection2Choice_Model
    if cls_name == 'im_choice':
        return im.Image_Choice_Model
    if cls_name == 'im_extract':
        return im.Image_Extract_Model
    else:
        LOG.error('No Class assigned to {}'.format(cls_name))
        raise NotImplementedError


def get_model(cfg):
    ''' Based on Trainer type of experiment, chooses or creates correct class '''
    ''' WARNING: Not ideal code design, must call _corr models before non-corr models'''
    trainer_name = cfg.trainer.name # trainer_name is composed of a base, and experiment suffixes
    if 'base_trainer' in trainer_name:
        return tm.MNL(cfg)
    elif 'lmnl_tab' in trainer_name:
        return tm.L_MNL(cfg)
    elif 'lmnl2_tab' in trainer_name:
        return tm.L_MNL2(cfg)
    elif 'embed_tab' in trainer_name:
        return tm.Embed_LMNL(cfg)
    elif 'tab_Imtester' in trainer_name:
        return tm.Tabular_ImTester(cfg)
    elif 'nn_tab' in trainer_name:
        return tm.Dense_NN(cfg)
    elif 'binary_tab' in trainer_name:
        return tm.BinaryModel(cfg)
    elif 'img_detect' in trainer_name:
        return im.Detection_Model(cfg)
    elif 'img_choice' in trainer_name:
        return im.Image_Choice_Model(cfg)
    elif ('choice_corr' in trainer_name) or ('extract_corr' in trainer_name):
        tab_class = get_class(cfg.trainer.tab_model)
        im_class = get_class(cfg.trainer.im_model)
        return im.ImCorr_Model(im_class, tab_class, cfg)
    # CAREFUL, due to nomenclature, added _corr experiments must be early in if-else statements
    elif ('choice_detect_corr' in trainer_name):
        tab_class = get_class(cfg.trainer.tab_model)
        multiImg_class = get_class(cfg.trainer.im_model)
        return im.ImCorrDetect_Model(multiImg_class, tab_class, cfg)
    elif 'choice_detect' in trainer_name:
        return im.ChoiceWithDetection_Model(cfg)
    elif 'detect2choice' in trainer_name:
        return im.Detection2Choice_Model(cfg)
    elif 'encoder_corr' in trainer_name:
        tab_class = get_class(cfg.trainer.tab_model)
        return im.ImCorrDetect_Model(im.Encoder_Model, tab_class, cfg)
    elif 'encoder' in trainer_name:
        return im.Encoder_Model(cfg)
    elif 'reversal_corr' in trainer_name:
        tab_class = get_class(cfg.trainer.tab_model)
        return im.ImCorrDetect_Model(im.Encoder_Reversal, tab_class, cfg)
    elif 'reversal' in trainer_name:
        return im.Encoder_Reversal(cfg)
    elif 'latent_corr' in trainer_name:
        tab_class = get_class(cfg.trainer.tab_model)
        return im.ImCorrDetect_Model(im.Latent_Choice_Model, tab_class, cfg)
    elif 'latent' in trainer_name:
        return im.Latent_Choice_Model(cfg)
    elif 'contrast_corr' in trainer_name:
        tab_class = get_class(cfg.trainer.tab_model)
        return im.ImCorrContrastive_Model(im.Contrastive_Model, tab_class, cfg)
    elif 'contrast' in trainer_name:
        return im.Contrastive_Model(cfg)
    elif 'gain' in trainer_name:
        return im.Gain_Model(cfg)
    else:
        LOG.error('No Models assigned to {}'.format(trainer_name))
        raise NotImplementedError


def get_backbone(cfg):
    ''' Return pretrained backbone as well as the layer to  change for  num class adaptation '''
    backbone = cfg.trainer.backbone_model
    if backbone == 'Res18':
        model = vis.resnet18(pretrained=True)
        linear_layer = 'fc'
        lay_idx = None
    elif backbone == 'Res50':
        model = vis.resnet50(pretrained=True)
        linear_layer = 'fc'
        lay_idx = None
    elif backbone == 'AlexNet':
        model = vis.alexnet(pretrained=True)
        linear_layer = 'classifier'
        lay_idx = 6
    else:
        LOG.error('No Backbones assigned to {}'.format(backbone))
        raise NotImplementedError

    return model, linear_layer, lay_idx


def get_saliency_layer(cfg):
    ''' Return layer name for saliency visualization: '''
    backbone = cfg.trainer.backbone_model
    if backbone == 'Res18':
        saliency_layer = 'layer4'
    elif backbone == 'AlexNet':
        saliency_layer = 'features'
    else:
        LOG.error('No Saliency Layers assigned to {}'.format(backbone))
        raise NotImplementedError
    return saliency_layer

def set_parameter_requires_grad(model, requires_grad):
    ### https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    for param in model.parameters():
        param.requires_grad = requires_grad
