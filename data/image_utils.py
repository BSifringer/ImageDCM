import torch
import torchvision.transforms as transforms
import cv2
import os
import enum
import logging
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import data.data_utils as dut
#https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder

from typing import List, Optional, Tuple, Union
from torch import Tensor
import math

LOG = logging.getLogger(__name__)

###  MAX NUMBER OF PASSANGERS OR PEDSTRIANS = 5 (Two loading Methods, create coordinates from small tiles, or create 1 img per Pedestrian coordinate)


def cv2_to_torch(img):
    # CV2 standards: BGR channels and (height,width,channels) dimensions (numpy object)
    # pytorch standards: RGB, (channels, height, width)
    img[:, :, :] = np.flip(img[:, :, :], -1)  # to RGB
    return torch.from_numpy(img.transpose((2, 0, 1)))


def torch_to_pyplot(img):
    return np.array(img.permute((1, 2, 0))).astype('float')


def load_tiles(tiles_folder):
    ''' Returns a dict of tilenames with their Tensor values '''
    ''' Note the 4th channel is used as a mask for processing and not as alpha '''
    tiles = {}
    for filename in os.listdir(tiles_folder):
        LOG.debug(f'Loading from file {os.path.join(tiles_folder, filename)}')
        img = cv2.imread(os.path.join(tiles_folder, filename))
        if img is not None:
            img = cv2_to_torch(img)
            mask = torch.all(img == 0, dim=0, keepdim=True) # Black (000) RGB now has 1 in 4th channel, used in add_image
            img_4channel = torch.cat((img, mask), dim=0)
            tiles.update({filename[:-4]: img_4channel})
            # tiles.update({filename[:-4]: img})
    return tiles


def cols_to_tiles(X_cols):
    """ Takes a list of used Variables.
    Returns the full dict for any variable to tile-filename core + list of core names from used variables
    """
    # fileNames = fold_tiles_dict.keys()
    # for fileName in fileNames:
    #     if fileName not in FileNames.default:
    #         LOG.warn(fileName + ' not found in default List')

    var_list = dut.UsedAttributes['all'].value[0] # keep order of attributes based on full list

    # Most Tile filenames are lower case of Tabular columns
    var_to_tile = {var: var.lower() for var in var_list}
    # List of name exceptions:
    var_to_tile.update({'CrossingSignal': 'trafficlight'})
    var_to_tile.update({'Barrier': 'trafficbarrier'})
    var_to_tile.update({'LargeMan': 'fatman'})
    var_to_tile.update({'LargeWoman': 'fatwoman'})
    var_to_tile.update({'MaleExecutive': 'businessman'})
    var_to_tile.update({'FemaleExecutive': 'businesswoman'})
    LOG.debug('columns received in cols_to_tiles {}'.format(X_cols))
    ordered_tile_variables = [var_to_tile[X_col] for X_col in X_cols]

    return var_to_tile, ordered_tile_variables


def add_image(im1, im2, coordinates):
    ''' Inplace transform version, im1 is a background to im2 '''
    ''' Before, images had 3 channels, a negative mask had to be created to avoid overwriting the background with black pixels'''
    ''' Now, images have 4 channels, the last channel is used as the negative mask (= 1 when rgb is black) '''
    # temp = im1.copy()
    temp = im1
    h = im2.shape[1]
    w = im2.shape[2]
    x, y = coordinates
    u = int(w/2)
    w = w-u
    # negative = (im2 == 0)
    # negative = torch.all(im2 == 0, dim=0, keepdim=True) # Returns True if all 3 channels are 0,0,0 (Black). Bools = 1xHxW
    # LOG.debug(
    #     f'temp_window : {temp[:, y-h:y, x-u:x+w].shape} negative: {negative.shape} h: {h} w: {w*2}')

    ''' with jitter and resize, issues with off limits coordinates. Fixing here to always fit in temp: '''
    if x-u < 0:
        add = -x+u
        u = x
        w = w+add
    if x+w > temp.shape[2]:
        add = x+w-temp.shape[2]
        w = temp.shape[2]-x
        u = u+add
    if y-h < 0:
        y = h
    temp[:, y-h:y, x-u:x+w] = torch.where(im2[-1:,...] == 1, temp[:, y-h:y, x-u:x+w], im2) # Only add im2 where mask is not 1
    return temp



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Add gaussian noise to tensor where values > 0
        return tensor + ((torch.randn(tensor.size()) * self.std)*255 + self.mean).to(tensor.dtype) * (tensor>0)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def gaussian_noise(img, sigma=1):
    return img + (sigma * torch.randn(img.size())*255).to(img.dtype)


class Tile_Manipulator():
    """An important component of the Image Dataset classes.  It is a class of itself to facilitate information accross functions.
    The functions are limited to  utilities such as loading tiles, sending coordinates or indices of used tiles. The tiles themselves belong to Image Datasets.
    """

    def __init__(self, X_cols, jitter=0, add_empty=False):
        # self.fold_tiles_dict = None  # updated when reading tiles
        # Define all scene variables in your dataset ( in the filename spelling )
        self.scene_variables = ['intervention',
                                'trafficbarrier', 'trafficlight']

        _, self.variables = cols_to_tiles(X_cols)
        self.scene_attributes = [
            var for var in self.variables if var in self.scene_variables]

        LOG.debug(self.variables)
        LOG.debug(self.scene_attributes)
        ''' Coordinates to create synthetic images - manually defined '''
        self.coordinates = {
            'ped_coordinates_left': [
                (80, 674), (142, 674), (210, 674), (130, 733), (191, 733)
                ],
            'ped_coordinates_right': [
                (305, 674), (373, 674), (438, 674), (333, 733), (397, 733)
                ],
            'passenger_coordinates': [
                (194, 113), (125, 63), (136, 113), (164, 63), (204, 63)
                ],
            'barrier_left': [(150, 611)],
            'barrier_right': [(361, 611)],
            'lights_sides': [(20, 692), (493, 692)],
            'lights_left': [(20, 726), (242, 726)],
            'lights_right': [(271, 658), (493, 658)],
            'trafficisland': [(256, 755)]
        }

        self.jitter = jitter # jitter coordinates of tiles if > 0
        self.add_empty = add_empty # Randomly add empty boxes when leftover coordinates are available


    def create_coordinates(self, X):
        ''' paralellized version of create_coordinates_single'''
        inputs = X.shape[0]
        # LOG.debug('X is shape of {}'.format(X.shape))
        if inputs > 1:
            # LOG.debug('Parallelizing batch of {}'.format(inputs))
            num_cores = multiprocessing.cpu_count()
            results = Parallel(n_jobs=num_cores)(
                delayed(self.create_coordinates_single)(X[i], self.jitter) for i in range(inputs))
        else:
            results = self.create_coordinates_single(X[0], self.jitter)
        return results

    def create_coordinates_single(self, X, jitter=0):
        """Return Coordinates from a 2 line scene description. All items are added in a dict with a list of coordinate tuples to associated key.
        Exception to charactesr, which are all in this format to keep correct image building order :  'characters':  [('char', (x,y)), ('char2', (x,y)), ... ]
        :param type X:  2xN_vars
        :return: list of 2 dicts containing key and coordinates of scene objects
        """
        # LOG.debug(f'Input shape X: {X.shape}')
        if jitter==0:
            coordinates = self.coordinates
        else:
            coordinates = {key: [(x+np.random.randint(-jitter,+jitter), y+np.random.randint(-jitter,jitter))
                                 for (x,y) in value] for key, value in self.coordinates.items()}
        scene = X.clone()
        tile_positions = [{}, {}]
        side_extensions = ['_left', '_right']
        barrier_scenario = 0
        lights = {}
        lights_color = {-1: '_red', 1: '_green'}
        # Get Lefthand and Rightside information
        for idx, (side, cols) in enumerate(zip(tile_positions, scene)):
            side_extension = side_extensions[idx]
            people = 0
            permuted_positions = np.random.permutation(
                len(coordinates['ped_coordinates_left'])).astype('int')
            characters = []
            for idx, var in enumerate(cols):
                # If barrier, choose from passenger coordinates
                # LOG.debug(f'{self.variables[idx]} : {var}')
                if idx < len(self.scene_attributes):
                    if self.variables[idx] == 'trafficbarrier':
                        if var != 1:
                            pos_people = 'ped_coordinates'+side_extension
                            people_varextension = '_walking'
                        else:
                            pos_people = 'passenger_coordinates'
                            people_varextension = '_passenger'
                            barrier_scenario = 1
                            side.update(
                                {self.variables[idx]: coordinates['barrier'+side_extension]})
                # Lights depend of both sides, create at the end of loop:
                    if self.variables[idx] == 'trafficlight':
                        if var != 0:
                            lights.update({side_extension: int(var)})
                else:
                    # Add characters to list, based on number
                    while var > 0:
                        var = var-1
                        characters.append(
                            self.variables[idx]+people_varextension)
                        people += 1
            if self.add_empty: # add "empty" characters 
                if people < len(coordinates['ped_coordinates_left']):
                    random_add = np.random.randint(
                        len(coordinates['ped_coordinates_left'])-people+1)
                    for i in range(random_add):
                        characters.append('empty'+people_varextension)
                        people += 1
            for i in range(len(coordinates['ped_coordinates_left'])-people):
                # Fill list with None, for permutation of characters  (we want to add positions in order for correct overlapping)
                characters.append(None)
            characters = np.array(characters)[permuted_positions]
            side.update({'characters': []})
            for pos, char in enumerate(characters):
                if char is not None:
                    side['characters'].append(
                        (char, coordinates[pos_people][pos]))
        # Create final full scene variables:
        if (barrier_scenario == 0) and (len(lights) == 2): # only a traffic island when no barrier and 2 lights
            tile_positions[0].update(
                {'trafficisland': coordinates['trafficisland']})
            # since we have 2 dicts, it is no issue, but check here in case
            assert lights_color[lights['_left']
                                ] != lights_color[lights['_right']]
            tile_positions[0].update(
                {'trafficlight'+lights_color[lights['_left']]: coordinates['lights_left']})
            tile_positions[1].update(
                    {'trafficlight'+lights_color[lights['_right']]: coordinates['lights_right']})
        elif len(lights):
            tile_positions[0].update(
                    {'trafficlight'+lights_color[list(lights.values())[0]]: coordinates['lights_sides']})

        return tile_positions


class MultiIdx_Tile_Manipulator(Tile_Manipulator):
    ''' Need implementation if quicker cpu processing is necessary '''
    def __init__(self):
        super(MultiIdx_Tile_Manipulator, self).__init__()
        self.ped_attributes = None

    def return_indices(self, X):
        # Batch size x 2 x n_vars
        return 0

    def return_coordinates(self, X):
        ped_coordinates = np.random.permutation(
                    len(self.ped_attributes))
        return 0


class CustomResize(transforms.Resize):
    ''' Custom Resize class to allow for random scaling of images, simply using scale factors of original image size'''
    def __init__(self, scales):
        size=1 # dummy for init of original class
        super(CustomResize, self).__init__(size, antialias=True) # add antialias to remove warning in torch update
        self.scales = scales

    def forward(self, img):
        img_height = img.shape[-2]
        img_width = img.shape[-1]
        # sample uniform in scales:
        scale_h = np.random.uniform(self.scales[0], self.scales[1])
        scale_w = np.random.uniform(self.scales[0], self.scales[1])

        self.size = (int(img_height*scale_h), int(img_width*scale_w))

        # return super(CustomResize, self).forward(img)

        # test cv2 resize: (more efficient for our case than origianl resize class (no antialias))
        arrImg = img.numpy().transpose(1, 2, 0)
        arrImg = cv2.resize(arrImg, (self.size[1], self.size[0]))
        tensorImg = torch.from_numpy(arrImg.transpose(2, 0, 1))
        return tensorImg


class CustomRandomErasing(transforms.RandomErasing):
    # Custom Version fixing silent erasing failure
    # From RandomErasing source code https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomErasing
    """Randomly selects a rectangle region in a torch.Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.PILToTensor(),
        >>>   transforms.ConvertImageDtype(torch.float),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])
    """


    @staticmethod
    def get_params(
        img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float], value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Changing this silent failure from original github code!!!:
        # # Return original image
        # return 0, 0, img_h, img_w, img
        ' Instead of silent fail, take a dimension, use scale to define mask value, then stay true to image scale area'
        ' NOTE: ratio values are thus ignored, but warning is raised'
        dim_select = np.random.randint(2)
        l = img_h if dim_select == 0 else img_w
        scale_select = torch.empty(1).uniform_(scale[0], scale[1]).item()
        dim_scale = torch.empty(1).uniform_(scale_select, 1).item()
        l = l * dim_scale
        l_dim2 = int(round(area*scale_select/l))
        l = int(round(l))
        (h, w) = (l, l_dim2) if dim_select == 0 else (l_dim2, l)
        if not (h <= img_h and w <= img_w):
                raise ValueError('Random Erasing code failed somehow, check the math, h={} w={} img_h={} img_w={}'.format(h, w, img_h, img_w))
        if value is None:
            v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
        else:
            v = torch.tensor(value)[:, None, None]
        if h/w < ratio[0] or h/w > ratio[1]:
            LOG.warning('Random Erasing Aspect Ratio outside of desired range due to scale selection. Aspect Ratio: {}, h:{}, w:{}'.format(h/w,h,w))
        
        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        return i, j, h, w, v
