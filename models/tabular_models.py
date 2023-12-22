import torch
import torch.nn as nn
import torch.nn.functional as F
from models import model_utils as mut

class MNL(nn.Module):
    def __init__(self, cfg):
        super(MNL, self).__init__()

        self.param = cfg.trainer.param
        # bias in not an ASC (same for all utilities), ASC captured by "intervention" variable:
        self.utility = nn.Linear(self.param.n_betas, 1, bias=False)

    def forward(self, X):
        return {'choice' : self.utility(X)}

    def return_r_layer(self):
        '''Tabular models have no image representation terms'''
        return None

    @staticmethod
    def preprocess_shapes(outputs, labels):
        ''' Turn one hot labels to class encoding, views for CCE loss '''
        labels['choice'] = labels['choice'].view(-1, 2).argmax(-1)
        outputs['choice'] = outputs['choice'].view(-1, 2)
        return outputs, labels

    @staticmethod
    def scoring(outputs, labels):
        ''' Score is defined here as a count before it is divided by batch size for obtaining accuracy'''
        return {'choice': (torch.max(outputs['choice'], 1)[1].view(
            labels['choice'].size()) == labels['choice']).sum()}

    def freeze(self, freeze_layers=True):
        ''' Classes Inheriting from MNL will freeze all parameters when calling freeze '''
        # True will freeze, False will unfreeze
        for param in self.parameters():
            param.requires_grad = not freeze_layers

    @staticmethod
    def load_optim_state_dict(optimizer, previous_optim_state):
        ''' Load optimizer state dict '''
        optimizer.load_state_dict(previous_optim_state)
        return

    def special_log(*args):
        ''' Log elements specific to the model'''
        pass

class NN_tab(MNL):
    def __init__(self, cfg):
        super(Dense_NN, self).__init__(cfg)
        self.n_layers = self.param.n_layers
        self.n_neurons = self.param.n_neurons
        self.dropout_prob = self.param.dropout_prob
        self.first = nn.Linear(self.param.n_betas, self.n_neurons)
        self.linears = nn.Sequential()
        for i in range(self.n_layers-1):
            self.linears.add_module('linear{}'.format(
                i+1), nn.Linear(self.n_neurons, self.n_neurons))
            self.linears.add_module("softplus{}".format(i+1), nn.Softplus())
            self.linears.add_module("dropout{}".format(
                i+1), nn.Dropout(self.dropout_prob))
        self.last = nn.Linear(self.n_neurons, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, X):
        # Representation Term
        r = F.softplus(self.first(X))
        r = self.dropout(r)
        r = self.linears(r)
        r = self.last(r)
        return {'choice': r}


''' NOTE: Trainer Support for the following models is now depricated due to lack of need in experiments and big Refactor for Evaluation compatibility:
    The use of automatic Heatmap packages such as Captum does not allow model.forward functions with unused inputs. (thus MNL.forward(X,Z) became MNL.forward(X)) '''


class L_MNL(MNL):
    ''' Classical LMNL using extra input Z - not used in our experiments'''
    def __init__(self, cfg):
        super(L_MNL, self).__init__(cfg)
        self.n_layers = self.param.n_layers
        self.n_neurons = self.param.n_neurons
        self.dropout_prob = self.param.dropout_prob
        self.first = nn.Linear(self.param.n_Z, self.n_neurons)
        self.linears = nn.Sequential()
        for i in range(self.n_layers-1):
            self.linears.add_module('linear{}'.format(
                i+1), nn.Linear(self.n_neurons, self.n_neurons))
            self.linears.add_module("relu{}".format(i+1), nn.ReLU())
            self.linears.add_module("dropout{}".format(
                i+1), nn.Dropout(self.dropout_prob))
        self.last = nn.Linear(self.n_neurons, 1, bias=False)
        # self.lin_z = nn.Linear(self.param.n_Z, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, X, Z):
        # Representation Term
        r = F.relu(self.first(Z))
        r = self.dropout(r)
        r = self.linears(r)
        r = self.last(r)
        # r = self.last(r) + self.lin_z(Z)
        #Utility:
        x = self.utility(X)
        return {'choice': x + r}
        


class Embed_LMNL(L_MNL):
    def __init__(self, cfg):
        super(Embed_LMNL, self).__init__(cfg)
        self.embed_layer = nn.Linear(
            self.n_neurons, self.param.n_betas, bias=False)
        # self.last = nn.Linear(self.n_neurons, self.param.n_betas, bias=True)
        self.merge_layer = nn.Linear(self.param.n_betas, 1, bias=False)

    def forward(self, X, Z):
        # Representation Term
        r = F.relu(self.first(Z))
        r = self.dropout(r)
        r = self.linears(r)
        q = self.last(r)
        #Utility:
        x = self.utility(X)

        emb = self.embed_layer(r)

        return {'choice': x + self.merge_layer(X*emb) + q }

class L_MNL2(L_MNL):
    ''' Uses only one input set ( similar to residual model )'''
    def __init__(self, cfg):
        super().__init__(cfg)
        self.first = nn.Linear(self.param.n_betas, self.n_neurons)

    def forward(self, X):
        # Representation Term
        r = F.relu(self.first(X))
        r = self.dropout(r)
        r = self.linears(r)
        r = self.last(r)
        #Utility:
        x = self.utility(X)
        return {'choice': x + r }


class Multiply_by_scalar(nn.Module):
    ''' Creates a single learnable scalar as a usable pytorch module'''
    def __init__(self,):
        super(Multiply_by_scalar, self).__init__()
        self.scalar = nn.Parameter(torch.tensor(1.))

    def forward(self, X):
        return X*self.scalar

class Return_Gru(nn.Module):
    def forward(self, first_arg, *args):
        return (first_arg[0][:,-1:,:]).swapaxes(1,2) #return last hidden state

class Tabular_ImTester(MNL):
    ''' This model is used for testing the image model using only tabular data. Image representation is replace with values of X_true'''
    ''' Note: used to screen methods quicker, however not used in final experiments, and results may not be generalizable '''
    def __init__(self,cfg):
        super().__init__(cfg)
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


        self.encParam = cfg.trainer.encParam
       
        self.added_latent_dim = self.encParam.added_latent_dim * self.encParam.latent_dim_multiplier
        self.num_latent_space = self.param.n_betas * self.encParam.latent_dim_multiplier + self.added_latent_dim
     


        # self.feature_space = nn.Linear(self.n_neurons, self.param.n_betas, bias=False)
        self.feature_space = nn.Linear(self.n_neurons, self.num_latent_space, bias=False)
        self.last = nn.Linear(self.param.n_betas, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)



       ### Layer from latent 2 detect
        if self.encParam.detect_layer == 'linear':
            self.detection_layer = nn.Linear(int(self.num_latent_space-self.added_latent_dim), int(self.param.n_betas))
        elif self.encParam.detect_layer == 'sequential':
            self.detection_layer = nn.Sequential(
                    nn.Linear(int(self.num_latent_space-self.added_latent_dim), int(self.param.n_betas)*self.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.param.n_betas)*self.encParam.seq_scale, int(self.param.n_betas)))
        elif self.encParam.detect_layer == 'identity':
            self.detection_layer = nn.Identity()
        else: raise NotImplementedError

        ### Layer from latent with no detect - to choice
        if self.encParam.choice_layer == 'linear':
            self.choice_layer = nn.Linear(int(self.added_latent_dim), 2, bias=False)
        elif self.encParam.choice_layer == 'sequential':
            self.choice_layer = nn.Sequential(
                    nn.Linear(int(self.added_latent_dim), int(self.param.n_betas)*self.encParam.seq_scale),
                    nn.ReLU(),
                    nn.Linear(int(self.param.n_betas)*self.encParam.seq_scale, 2, bias=False))
        else: raise NotImplementedError


        #Activation from feature to latent space
        if self.encParam.activation == 'tanh':
            self.activation = nn.Tanh()
        elif self.encParam.activation == 'relu':
            self.activation = nn.ReLU()
        elif self.encParam.activation == 'identity':
            self.activation = nn.Identity()
        elif self.encParam.activation == 'GRU':
            self.activation = nn.Sequential(nn.GRU(2, 2, num_layers=4, batch_first=True), Return_Gru())
            self.choice_layer = nn.Linear(2, 2, bias=False)
        else: raise NotImplementedError

        self.scalar_multiply = Multiply_by_scalar()



        self.freeze_dcm = cfg.trainer.freeze_dcm

    def forward(self, X, X_true):
        r = F.relu(self.first(X_true))
        r = self.dropout(r)
        r = self.linears(r)
        features = self.feature_space(r)
        detected, invariant = self.detection_layer(features[...,:-self.added_latent_dim]), self.activation(features[...,-self.added_latent_dim:].swapaxes(1,2))

        # latent = self.last(detected) + self.scalar_multiply(invariant)
        latent = self.last(detected) + (invariant)
        utility_weights = self.utility.weight
        last_weights = self.last.weight
        weight_loss = torch.norm(utility_weights.detach()-last_weights)
        # true_masked = X_true[]
        #Utility:
        x = self.utility(X)

        return {'choice': x + self.scalar_multiply(invariant.detach()), 'detected': detected, 'latent': latent, 'beta': weight_loss, 'representation': invariant.squeeze()}

    @staticmethod
    def preprocess_shapes(outputs, labels):
        ''' This function takes into account self.forward outpout and trainer's inference function (e.g. Choice_Detection_trainer) '''
        # if self.fuse_detection_labels:
        #     labels['X'] = labels['X'].sum(dim=-2)
        outputs['detected'] = outputs['detected'].reshape(outputs['detected'].shape[0], -1)
        labels['X'] = labels['X'].reshape(outputs['detected'].shape)
        labels['choice'] = labels['choice'].view(-1, 2).argmax(-1)
        outputs['choice'] = outputs['choice'].view(-1, 2)
        outputs['latent'] = outputs['latent'].view(-1, 2)
        return outputs, labels



    @staticmethod # Copied from image models
    def scoring(outputs, labels):
        """ 1 attribute can have multiple counts. The accuracy is multiplied by the batch size for (score) correct averaging by main loop"""
        img_utility, detected  = outputs['choice'].detach(), outputs['detected'].detach()
        label_choice, X = labels['choice'], labels['X']
        detectables = torch.abs(X).sum()
        diff = torch.abs(torch.round(detected) - X).sum()
        return {'detect': (detectables-diff) / detectables * len(X),
                'choice': (torch.max(img_utility, 1)[1].view(label_choice.size()) == label_choice).sum()}

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=False)
    
    def freeze(self, freeze_layers=True):
        # True will freeze, False will unfreeze
        requires_grad = not freeze_layers
        if not self.freeze_dcm:
            for name, layer in self.named_modules():
                if 'utility' not in name:
                    mut.set_parameter_requires_grad(layer, requires_grad=requires_grad)
        else:
            mut.set_parameter_requires_grad(
                self.utility, requires_grad=requires_grad)



class Dense_NN(MNL):
    def __init__(self, cfg):
        super(Dense_NN, self).__init__(cfg)
        self.n_layers = self.param.n_layers
        self.n_neurons = self.param.n_neurons
        self.dropout_prob = self.param.dropout_prob
        self.first = nn.Linear(
            self.param.n_Z+self.param.n_betas, self.n_neurons)
        self.linears = nn.Sequential()
        for i in range(self.n_layers-1):
            self.linears.add_module('linear{}'.format(
                i+1), nn.Linear(self.n_neurons, self.n_neurons))
            self.linears.add_module("softplus{}".format(i+1), nn.Softplus())
            self.linears.add_module("dropout{}".format(
                i+1), nn.Dropout(self.dropout_prob))
        self.last = nn.Linear(self.n_neurons, 1, bias=False)
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, X, Z):
        # Representation Term
        r = F.softplus(self.first(torch.cat([X, Z], axis=-1)))
        r = self.dropout(r)
        r = self.linears(r)
        r = self.last(r)
        return {'choice': r}


class BinaryModel(nn.Module):
    ''' Model to verify correct implementation of MNL. This model with BCE loss produces same results as MNL with CCE loss'''
    def __init__(self, cfg):
        super(BinaryModel, self).__init__()

        self.param = cfg.trainer.param
        # bias in not an ASC (same for all utilities), ASC captured by "intervention" variable:
        self.utility = nn.Linear(self.param.n_betas, 1, bias=False)

    def forward(self, X, Z):
        U = X[:, 1]-X[:, 0]
        return {'chocie': self.utility(U)}

    @staticmethod
    def preprocess_shapes(outputs, labels):
        ''' U2-U1 is the binary utility. So label U2 is the choice YES/No, Views for BCE loss '''
        labels['choice'] = labels['choice'][:, 1]
        outputs['choice'] = outputs['choice'].view(-1)
        return outputs, labels

    @staticmethod
    def scoring(outputs, labels):
        ''' Score is defined here as a count before it is divided by batch size for obtaining accuracy'''
        return {'bce:'(labels['choice'] == torch.round(torch.sigmoid(outputs['choice']))).sum()}

    def special_log(*args):
        ''' Log elements specific to the model'''
        pass