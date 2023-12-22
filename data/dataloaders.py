#Credits for FastTensorDataLoader: https://github.com/hcarlens/pytorch-tabular
import torch
from torchvision import transforms
import pandas as pd
import numpy as np
from omegaconf import open_dict
import logging

import data.data_utils as dut
import data.image_utils as iut
import data.masking as msk

''' This file contains functions which assembles datasets with dataloaders as well as their defined classes '''
''' NOTES: PairwiseTabular 10x slower data loading;  but 2x faster getitems for tabular data if memory-size fits RAM in float32'''
''' Main project and many experiments requires Masking. This is efficient for getitem, thus datasets have integrated maskers '''

LOG = logging.getLogger(__name__)


def getDataloaders(cfg):
    """Based on cfg arguments, builds the correct dataset with optimal dataloaders (train/val/test)
    """
    method = cfg.data.dataloader
    masker = None
    # Batch_size on eval needs to be smaller for cuda memory issue when getting Hessian
    batch_multipliers = [1, 1, cfg.trainer.test_batchsize_multiplier]
    if method == 'PairwiseTabular':
        loaded_vars = dut.UsedAttributes[cfg.data.varChoice].value
        if loaded_vars[1]: # If Z_cols is not empty
            LOG.error('You are loading Z columns from the old data mangement for running old LMNL models where Z was Individual specific variables.\
                       Now, Z is a subset of X for Final Project. Recode new behavoirs if needed \n {}'.format(loaded_vars[1]))
            raise NotImplementedError
        Z_cols = cfg.trainer.masked_tabular_list if not cfg.trainer.masked_tabular_list is None else []
        loaded_vars = (loaded_vars[0],Z_cols)
        XZY_sets, used_columns = dut.pandas2subsets(cfg, columns=loaded_vars)

        # Update cfg with number of used variables:
        update_cfg(cfg, used_columns)
        dataloaders = []
        for subset_XZY, scale_batch in zip(XZY_sets, batch_multipliers):
            dataset = SurveyDataset(subset_XZY)
            # Fast speeds when sampler loads batches at a time, and loader calls sampler once only
            fastsampler = FastSampler(dataset, batch_size=int(cfg.trainer.batch_size*scale_batch),
                                      drop_last=cfg.trainer.drop_last, shuffle=cfg.trainer.shuffle)
            dataloader = torch.utils.data.DataLoader(
                              dataset, batch_size=None, sampler=fastsampler,
                              num_workers=cfg.trainer.num_workers)
            dataloaders.append(dataloader)

    elif method == 'IndexTabular':
        loaded_vars = dut.UsedAttributes[cfg.data.varChoice].value
        if loaded_vars[1]: # If Z_cols is not empty
            LOG.error('You are loading Z columns from the old data mangement for running old LMNL models where Z was Individual specific variables.\
                       Now, Z is a subset of X for Final Project. Recode new behavoirs if needed \n {}'.format(loaded_vars[1]))
            raise NotImplementedError
        Z_cols = cfg.trainer.masked_tabular_list if not cfg.trainer.masked_tabular_list is None else []
        loaded_vars = (loaded_vars[0],Z_cols)
        df, indices, columns = dut.pandas2sindices(cfg, loaded_vars)
        update_cfg(cfg, columns)
        dataloaders = []
        for subset_indice, scale_batch in zip(indices, batch_multipliers):
            LOG.debug(f' Size of indices: {len(subset_indice)}')
            dataset = IndexSurveyDataset(
                df, subset_indice, columns, cfg.data.label)
            fastsampler = FastSampler(dataset, batch_size=int(cfg.trainer.batch_size*scale_batch),
                                      drop_last=cfg.trainer.drop_last, shuffle=cfg.trainer.shuffle)
            dataloader = torch.utils.data.DataLoader(
                              dataset, batch_size=None, sampler=fastsampler,
                              num_workers=cfg.trainer.num_workers)
            LOG.debug(
                f'Size of dataset {len(dataset)}, size of dataloader {len(dataloader)}')
            dataloaders.append(dataloader)

    elif method == 'IndexImage': # acceleration for tabular datasets are no longer required (Pairwise, fastsampler, etc )
        loaded_vars = dut.UsedAttributes[cfg.data.varChoice].value
        if loaded_vars[1]: # If Z_cols is not empty
            LOG.error('You are loading Z columns from the old data mangement for running old LMNL models where Z was Individual specific variables.\
                       Now, Z is a subset of X for Final Project. Recode new behavoirs if needed \n {}'.format(loaded_vars[1]))
            raise NotImplementedError
        Z_cols = cfg.trainer.masked_tabular_list if not cfg.trainer.masked_tabular_list is None else []
        loaded_vars = (loaded_vars[0],Z_cols)
        df, indices, columns = dut.pandas2sindices(cfg, loaded_vars)
        LOG.debug(columns)
        update_cfg(cfg, columns)
        dataloaders = []
        tile_manip = iut.Tile_Manipulator(cfg.data.X_cols, cfg.data.jitter, cfg.trainer.add_empty_tiles)
        masker = msk.Mask_Manipulator(cfg)
        # For PairwiseTabular style, only change the inherited class here
        for subset_indice, scale_batch in zip(indices, batch_multipliers):
            dataset = ImgDataset(IndexSurveyDataset, cfg, tile_manip, masker,
                                 df, subset_indice, columns, cfg.data.label)
            dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=int(
                        cfg.trainer.batch_size*scale_batch),
                    num_workers=cfg.trainer.num_workers)
            dataloaders.append(dataloader)
    else:
        raise NotImplementedError

    LOG.info(
        f'Train Dataloader {method} approx length: {dataloaders[0].__len__()*2*cfg.trainer.batch_size}\n'
        + f'Val Dataloader {method} approx length: {dataloaders[1].__len__()*2*cfg.trainer.batch_size}\n'
        + f'Test Dataloader {method} approx length: {dataloaders[2].__len__()*2*cfg.trainer.batch_size}')
    if not (dataloaders[1].__len__()) * (dataloaders[2].__len__()):
        LOG.error(
            'A test/val DataLoader is empty, this may happen when drop_last=true and batch_size > len(dataset)')
        raise ValueError
    if masker is None: # Masker is only used at this stage for image datasets
        masker = msk.Mask_Manipulator(cfg)
    return dataloaders, masker


def returnDatasets(cfg):
    ''' Get desired attributes, Reads data from csv returned as Datasets with final used variables'''
    loaded_vars = dut.UsedAttributes[cfg.data.varChoice].value
    [XZY_train, XZY_val, XZY_test], used_columns = dut.pandas2subsets(
        cfg, columns=loaded_vars)

    return [SurveyDataset(XZY_train), SurveyDataset(XZY_val), SurveyDataset(XZY_test)], used_columns


def update_cfg(cfg, columns):
    ''' Updates cfg with param and variable details '''
    with open_dict(cfg):
        X_cols, Z_cols = columns
        cfg.trainer.param.n_betas = len(X_cols)
        cfg.trainer.param.n_Z = len(Z_cols)
        cfg.data.X_cols = X_cols
        cfg.data.Z_cols = Z_cols
        LOG.info(f' X_cols count {cfg.data.X_cols}')
        LOG.info(f' Z_cols count {cfg.data.Z_cols}')


class FastSampler(torch.utils.data.sampler.Sampler):
    """ This sampler is inspired of FastTensorDataLoader and is custom for "sampler" of torch Dataloader
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    :param data_source: A torch Dataset upon which the sampler is based.
    :param int batch_size:  --
    :param bool drop_last:  True to remove remaining data too small for batch_size
    :param bool shuffle: Shuffle indices acessing the dataset (sampling)
    """

    def __init__(self, data_source: torch.utils.data.Dataset, batch_size: int,
                 drop_last: bool, shuffle: bool = False) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
#       self.data_source = data_source # in torch Docs, but  is it not unecessary?
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.dataset_len = len(data_source)

        if shuffle:
            self.indices = torch.randperm(self.dataset_len)
        else:
            self.indices = torch.tensor(range(self.dataset_len))

    def __iter__(self):
        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        idx = 0
        while True:
            batch = self.indices[idx
                                 * self.batch_size: (idx+1)*self.batch_size]
            if len(batch) == self.batch_size:
                yield batch
            else:
                if len(batch) and not self.drop_last:
                    yield batch
                else:
                    break
            idx += 1

    def __len__(self) -> int:
        if self.drop_last:
            return (self.dataset_len) // self.batch_size
        else:
            return ((self.dataset_len) + self.batch_size - 1) // self.batch_size


class IndexSurveyDataset(torch.utils.data.Dataset):
    """Pairwise getitem, To Use with pytorch Dataloader. Acess a pandas dataset for memory optimization
    """

    def __init__(self, df, indices, columns, lab_col):
        self.df = df
        self.indices = torch.tensor(np.array(indices)).view(-1).int()
        LOG.debug(f'Indices tensor shape {self.indices.size()}')
        self.X_cols = columns[0]
        self.Z_cols = columns[1]
        self.lab_col = lab_col

    def __getitem__(self, idx):  # With FastDataloader,  idx is a list
        if isinstance(idx, int):
            idx = np.array([idx])
        first_idx = self.indices[2*idx].tolist()
        second_idx = self.indices[2*idx+1].tolist()
        data1 = self.df.iloc[first_idx]
        data2 = self.df.iloc[second_idx]
        X = torch.stack([torch.from_numpy(data1[self.X_cols].values.astype('float')).view(-1, len(self.X_cols)), torch.from_numpy(
            data2[self.X_cols].values.astype('float')).view(-1, len(self.X_cols))], axis=1)
        Z = torch.stack([torch.from_numpy(data1[self.Z_cols].values.astype('float')).view(len(first_idx), len(self.Z_cols)), torch.from_numpy(
            data2[self.Z_cols].values.astype('float')).view(len(second_idx), len(self.Z_cols))], axis=1)
        labels = torch.stack([torch.from_numpy(np.array(
                    data1[self.lab_col])).view(-1, len(idx)), torch.from_numpy(np.array(data2[self.lab_col])).view(-1, len(idx))], axis=1)
        ''' Dataframe multiIndex needs .values, but single index not? '''
        # labels = torch.stack([torch.from_numpy(
        #     data1[self.lab_col].values), torch.from_numpy(data2[self.lab_col].values)], axis=1)

        return X.float(), Z.float(), labels.float()  #  shapes (idx_size, 2, n_vars)

    def __len__(self):
        return int(len(self.indices)/2)


class SurveyDataset(torch.utils.data.Dataset):
    """Pairwise getitem, To Use with pytorch Dataloader.  Init with X, Z and labels in a list
    """

    def __init__(self, XZY: list):
        self.X, self.Z, self.labels = XZY

    def __getitem__(self, idx):  # With FastDataloader,  idx is a list
        X = torch.stack([self.X[2*idx], self.X[2*idx+1]], axis=1)
        Z = torch.stack([self.Z[2*idx], self.Z[2*idx+1]], axis=1)
        labels = torch.stack(
            [self.labels[2*idx], self.labels[2*idx+1]], axis=1)
        return X, Z, labels  #  shapes (idx_size, 2, n_vars)

    def __len__(self):
        return int(len(self.X)/2)


def ImgDataset(cls, cfg, img_manipulator: iut.Tile_Manipulator, masker: msk.Mask_Manipulator, *args):
    ''' ImgDataset inherits a tabular dataset. This function returns the class based on desired inherited backbone '''
    class ImgDataset(cls):
        def __init__(self, cfg, img_manipulator: iut.Tile_Manipulator, masker, *args):
            super(ImgDataset, self).__init__(*args)
            self.cfg = cfg
            self.tiles = {}
            self.load_tiles()  # Get them from Memory, send them to device?!
            #  Image Utility class with it's own memory of used variables_
            self.img_manip = img_manipulator
            self.resize_factor = self.cfg.data.image_shrink_factor
            self.update_cfg()

            self.masker = masker

            self.noise_transform = transforms.Lambda(iut.gaussian_noise)
            self.noise_transform = iut.AddGaussianNoise(0., 0.1)
            self.color_transform = transforms.ColorJitter(brightness=.5, hue=.3)
            self.blur_transform = transforms.GaussianBlur((3,3), sigma=(0.1, 0.1)) # Blur does not seem to work on this size tiles
            self.flip_transform = transforms.RandomHorizontalFlip(1)
            if self.cfg.data.img_transforms == 'random':
                self.transforms = transforms.RandomApply([self.noise_transform,
                                                     self.color_transform,
                                                     self.blur_transform,
                                                     self.flip_transform], p=0.5)
            elif self.cfg.data.img_transforms == 'all':
                self.transforms = transforms.Compose([self.noise_transform,
                                                     self.color_transform,
                                                     self.blur_transform,
                                                     self.flip_transform])
            elif self.cfg.data.img_transforms == 'none':
                self.transforms = transforms.Compose([])
            else:
                raise NotImplementedError
            
            if cfg.trainer.tile_resize:
                self.tile_resize = transforms.Compose([iut.CustomResize((cfg.trainer.tile_resize_scale[0], cfg.trainer.tile_resize_scale[1]))])
            else:
                self.tile_resize = transforms.Compose([])

            if cfg.trainer.background_transforms:
                # Color transform on background is extremely slow (scales up the training time > x5), and not very useful
                # self.background_transforms = transforms.RandomApply([self.noise_transform,
                #                                         self.color_transform,
                #                                         self.blur_transform])
                self.background_transforms = transforms.Compose([self.noise_transform])
            else:
                self.background_transforms = transforms.Compose([])

            # If black masking is on, a chosen pixel value is set to black
            self.val_to_black = cfg.trainer.black_masking

 
        def load_tiles(self):
            # self.tiles = self.img_manip.read_tiles(self.cfg.data.tiles_folder)
            self.tiles = iut.load_tiles(self.cfg.data.tiles_folder)
            if self.cfg.data.build_img_gpu:
                LOG.warning('Building images on GPU may not implemented')
                using_cuda = torch.cuda.is_available() and self.cfg.trainer.cuda
                device = torch.device('cuda' if using_cuda else 'cpu')
                if not using_cuda:
                    LOG.warning("Set to build img on gpu, yet trainer not on cuda")
                for key, tile in self.tiles.items():
                    self.tiles[key] = tile.to(device)
            if self.cfg.trainer.add_empty_tiles: # create new tiles for empty walking and passenger, alpha channel is 1 (invisible)
                zero_tensor = torch.zeros_like(self.tiles['man_walking']).to(self.tiles['man_walking'].device)
                zero_tensor[-1,:,:] = 1
                self.tiles['empty_walking'] = zero_tensor
                zero_tensor = torch.zeros_like(self.tiles['man_passenger']).to(self.tiles['man_walking'].device)
                zero_tensor[-1,:,:] = 1
                self.tiles['empty_passenger'] = zero_tensor
            LOG.debug(f'Loaded tiles from folder: {self.tiles.keys()}')
            return 0

        def update_cfg(self):
            channels, height, width = self.tiles['L'].shape
            final_width = int(width/self.resize_factor)
            with open_dict(self.cfg):
                self.cfg.data.img_height = int(height*final_width / width)
                self.cfg.data.img_width = final_width

        def __getitem__(self, idx):
            X, Z, labels = super(ImgDataset, self).__getitem__(idx)
            X_img = self.masker.image_X_masking(X.clone().detach()) # Change tiles to be generated (not an image mask)
            tile_positions = self.img_manip.create_coordinates(X_img)

            if not self.cfg.data.build_img_gpu: # if built on gpu, the trainer must take this code, preferably removed for loops
                imgs = self.create_imgs(tile_positions)
                p = transforms.Compose(
                    [transforms.Resize(int(imgs.size()[-1]/self.resize_factor), antialias=True)])
                return X, Z, labels, p(imgs)
                #  shapes (idx_size, 2, n_vars) - keep same order as tabular dataloaders; add images
            else:
                return X, Z, labels, tile_positions

        def create_imgs(self, tile_positions):
            """Create a single image base on tile positions given in coordinates.
            NOTES: For now, create only leftside, without death flags (2 sides not needed)
            NOTES: Left and right images only differ in terms of arrow direction and death flags. All characters appear on both shown images.
            :param list_dict tile_positions:  left+right hand side dict of characters and positions
            :return: image as torch tensor
            :rtype: type
            """
            base_img = self.return_background(0)
            base_img[:-1] = self.background_transforms(base_img[:-1]) # transform on rgb channels only
            for side in tile_positions:
                for tile_name, coordinates in side.items():
                    if tile_name == 'characters': # characters have their own dict 
                        for char, coord in coordinates:
                            tile = self.tiles[char]
                            tile[:-1] = self.transforms(tile[:-1]) #  Apply random transforms on rgb channels only
                            tile = self.tile_resize(tile) #  Apply resize
                            tile = self.masker.mask_image(tile, char) #  Apply mask (condition on char)
                            base_img = iut.add_image(base_img, tile, coord)
                            # if self.val_to_black: # depricated, now using alpha channels for black or transparent masking
                            #     base_img =  self.masker.val_to_black(base_img)
                    else:
                        tile = self.tiles[tile_name] # background tiles are only a list of coordinates
                        for coord in coordinates:
                            # tile = self.transforms(tile) # do not apply transforms to lights, barriers,  ...
                            tile = self.masker.mask_image(tile, tile_name) #  Apply mask (condition on char)
                            base_img = iut.add_image(base_img, tile, coord)

            return base_img[:-1,...]/255.

        def return_background(self, side_idx: int):
            if side_idx == 0:
                return self.tiles['L'].clone()
            elif side_idx == 1:
                return self.tiles['R'].clone()
            
        def set_controlled_mask(self, scale, p=1):
            self.masker.set_controlled_mask(scale, p)
            return 0

    return ImgDataset(cfg, img_manipulator, masker, *args)