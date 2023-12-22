import enum
import pandas as pd
import torch
import dask.dataframe as dd
import pickle
import numpy as np
import logging
import data.image_utils as iut
import torchvision.transforms as transforms

LOG = logging.getLogger(__name__)


def remove_tabular(indices, X):
    X[:, ..., indices] = 0
    return X


def replace_tabluar(reduced_indices_list, target_indices, X):
    ''' This function turns reduced indices into chosen target (e.g., MaleDoctor becomes Man)'''
    ''' Unfortunately,  found no solution to vectorize multi-to-single indice addition'''
    for target_indice, reduced_indices in zip(target_indices, reduced_indices_list):
        X[:, ..., target_indice] += X[:, ..., reduced_indices].sum(dim=-1)
        X[:, ..., reduced_indices] = 0
    return X


def reduce_tabluar(reduced_indices_list, target_indices, X):
    ''' This function sums reduced indices over chosen target (e.g., MaleDoctor adds 1 to Man) (Doctor is thus a detail residual)'''
    for target_indice, reduced_indices in zip(target_indices, reduced_indices_list):
        X[:, ..., target_indice] += X[:, ..., reduced_indices].sum(dim=-1)
    return X

class Mask_Manipulator():
    ''' The mask manipulator performs masking and augmenting for tabular or image input by removing, replacing or reducing values in the tabular input.'''
    ''' Update: Now also has a controlled mask for image input. This is a random erasing mask that is applied to the image input. '''

    def __init__(self, cfg):
        self.masked_tiles = self.get_masked_tiles(cfg)
        self.masked_tabular = self.get_masked_tabular(cfg)
        using_cuda = torch.cuda.is_available() and cfg.trainer.cuda
        train_device = torch.device('cuda' if using_cuda else 'cpu')
        self.reduce_replace_dict = self.get_substitution_dict()
        self.indices=None

        self.tile_to_ind = self.get_tile_to_ind(cfg)

        ''' The following code prepares the masking functions. Place loops here if needed, so the final function can be vectorized '''
        if cfg.trainer.tabular_masking in ['remove','composed','erase']: # all these modes require removing tabular info. Then either nothing, mask or erase on image
            self.indices = np.array([np.where(np.array([cfg.data.X_cols]).flatten() == masked)
                                for masked in self.masked_tabular]).flatten()
            indices = torch.tensor(self.indices).to(train_device)

            def tabular_masking(X): return remove_tabular(indices, X)

        elif cfg.trainer.tabular_masking in ['replace', 'reduce']:
            ''' First turn Names of Variables into indices.
            Then organize them into clusters with a dict.
            Then turn them to list and array for masking function '''
            indices_to_substitute = np.array([np.where(np.array([cfg.data.X_cols]).flatten() == masked)
                                              for masked in self.masked_tabular]).flatten()
            ## If you get a KeyError, change get_substitution_dict
            targeted_values = [self.reduce_replace_dict[substituted]
                               for substituted in self.masked_tabular]
            target_indices = np.array([np.where(np.array([cfg.data.X_cols]).flatten() == masked)
                                       for masked in targeted_values]).flatten()
            target_substit_dict = {key: [] for key in target_indices}
            for key, value in zip(target_indices, indices_to_substitute):
                target_substit_dict[key].append(value)
            target_indices = list(target_substit_dict.keys())
            indices_to_substitute = [
                np.array(target_substit_dict[target]) for target in target_indices]
            if cfg.trainer.tabular_masking == 'replace':
                def tabular_masking(X):
                    return replace_tabluar(indices_to_substitute, target_indices, X)
            if cfg.trainer.tabular_masking == 'reduce':
                def tabular_masking(X):
                    return reduce_tabluar(indices_to_substitute, target_indices, X)
        elif cfg.trainer.tabular_masking is None:
            def tabular_masking(X): return X
        else:
            raise NotImplementedError


        ''' Now set the function of X: '''
        self.mask_tabular = tabular_masking

        self.erase_value = [0,0,0,1] # transparent masking
        if cfg.trainer.black_masking:
            # self.erase_value = [255,0,255] # Magenta
            self.erase_value = 0
            # self.erase_value = float('nan')



        ''' Either remove tiles completely or apply erasing transformation on special tiles'''
        self.controlled_mask = iut.CustomRandomErasing(p=cfg.data.controlled_mask_p,
                                                       scale=(cfg.data.controlled_mask_scale, cfg.data.controlled_mask_scale),
                                                       ratio=(0.05, 20), value=self.erase_value)
        self.mask_transform = transforms.Compose([])


        if cfg.trainer.image_masking is None:
            def image_X_masking(X): return X
            self.mask_transform = transforms.Compose([])
            self.erased_indices = None # transform will return tile untouched

        elif cfg.trainer.image_masking == 'masked_only':
            all_indices = np.arange(len(cfg.data.X_cols))
            leftover_indices = torch.tensor(all_indices[~np.isin(all_indices, self.indices)])
            def image_X_masking(X): return remove_tabular(leftover_indices, X) # Create image with masked tiles only, removing leftover tiles
            self.mask_transform = transforms.Compose([])
            self.erased_indices = None

        elif cfg.trainer.image_masking == 'erase':
            self.erased_indices = None # compose the transform, all indices are selected
            def image_X_masking(X): return X
            self.mask_transform = transforms.Compose([self.controlled_mask]) # CustomRandomErase transform all tiles

        elif cfg.trainer.image_masking == 'composed':
            all_indices = np.arange(len(cfg.data.X_cols))
            leftover_indices = torch.tensor(all_indices[~np.isin(all_indices, self.indices)])
            self.erased_indices = leftover_indices # Transform all tiles except tabular masked ones
            def image_X_masking(X): return X # Create image with all tiles
            self.mask_transform = transforms.Compose([self.controlled_mask])

        else:
            raise NotImplementedError
        self.image_X_masking = image_X_masking

        #add indice to erase for traffic island:
        if self.erased_indices is not None:
            self.erased_indices = torch.cat([self.erased_indices, torch.tensor([self.tile_to_ind['trafficisland']])])

    def mask_image(self, tile, char=None):
        ''' Called when building the image, each tile is assessed for masking based on experiment configuration'''
        if self.erased_indices is None: # For erase all, or erase none (transform is empty)
            return self.mask_transform(tile) 
        elif char is None: # catch None before testing it in the tile_to_ind dict
            return self.mask_transform(tile)
        elif 'empty' in char: # empty experiment is to add black boxes on empty tiles - only used when composed
            return self.mask_transform(tile)
        elif self.tile_to_ind[char] in self.erased_indices:
            return self.mask_transform(tile)
        return tile


    def set_controlled_mask(self, scale, p=1):
        self.controlled_mask = iut.CustomRandomErasing(p=p, scale=(scale, scale), ratio=(0.05, 20), value=self.erase_value)
        self.mask_transform = transforms.Compose([self.controlled_mask])


    def val_to_black(self, image):
        ''' Depricated - used to turn specific pixel colors to black'''
        mask = torch.all(image.permute(1,2,0) == torch.tensor(self.erase_value), dim=-1)
        image[:,mask] = 0
        return image

    def get_masked_tiles(self, cfg):
        masked_list = cfg.trainer.masked_tiles_list
        if masked_list is None:
            return []
        else:
            return np.array([masked_list]).flatten()

    def get_masked_tabular(self, cfg):
        ''' turn None into an empty list '''
        masked_list = cfg.trainer.masked_tabular_list
        if masked_list is None:
            return []
        else:
            return np.array([masked_list]).flatten()

    def get_substitution_dict(self):
        ''' By default, and as seen on visuals, single gendered personas are reduced to 'Man'
        Children and both Man and Woman have no current substition rule - by design'''
        return {
            'Pregnant': 'Woman',
            'OldMan': 'Man',
            'OldWoman': 'Woman',
            'Homeless': 'Man',
            'LargeWoman': 'Woman',
            'LargeMan': 'Man',
            'Criminal': 'Man',
            'MaleExecutive': 'Man',
            'FemaleExecutive': 'Woman',
            'FemaleAthlete': 'Woman',
            'MaleAthlete': 'Man',
            'FemaleDoctor': 'Woman',
            'MaleDoctor': 'Man'
        }


    def get_tile_to_ind(self, cfg):
        ''' Prepare tiles to indices, passengers are considered to be masked same as pedestrians '''
        tile_to_ind = {key: idx for idx, key in enumerate(iut.cols_to_tiles(cfg.data.X_cols)[1])}
        tile_passenger_to_ind = {key+'_passenger': idx for idx, key in enumerate(iut.cols_to_tiles(cfg.data.X_cols)[1])}
        tile_walking_to_ind = {key+'_walking': idx for idx, key in enumerate(iut.cols_to_tiles(cfg.data.X_cols)[1])}
        tile_to_ind.update(tile_passenger_to_ind)
        tile_to_ind.update(tile_walking_to_ind)
        new_idx = len(cfg.data.X_cols)
        tile_to_ind['trafficisland'] = new_idx
        # Update traffic lights:
        tile_to_ind['trafficlight_red'] = tile_to_ind['trafficlight']
        tile_to_ind['trafficlight_green'] = tile_to_ind['trafficlight']
        return tile_to_ind
