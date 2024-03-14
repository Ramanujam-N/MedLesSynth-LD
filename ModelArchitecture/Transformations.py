import torch
from scipy.ndimage import rotate
from skimage.exposure import rescale_intensity
import numpy as np
from torchvision import transforms

################################################ FOR 2D Transformations ##########################################################################3##############

class RandomRotation(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        #print(sample['input'].shape)
        if len(sample['input'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        angle = self.get_params(self.degrees)
        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)
        
        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.
        for sh in range(input_data.shape[2]):
            input_rotated[:, :, sh, 0] = rotate(input_data[:, :, sh, 0], float(angle), reshape=False, order=self.order,
                                                mode='nearest')

            if self.labeled:
                gt_rotated[:, :, sh, 0] = rotate(gt_data[:, :, sh, 0], float(angle), reshape=False, order=self.order,
                                                 mode='nearest')
                gt_rotated[:, :, sh, 1] = rotate(gt_data[:, :, sh, 1], float(angle), reshape=False, order=self.order,
                                                 mode='nearest')
                gt_rotated = (gt_rotated > 0.6).astype(float)

        # Update the dictionary with transformed image and labels
        rdict['input'] = input_rotated
       
        if self.labeled:
            rdict['gt'] = gt_rotated
        sample.update(rdict)
        return sample

class RandomContrastMatching(object):
    def __init__(self):
        pass
    def __call__(self,sample):
        
        rdict = {}
        input_data = sample['input']
        input_data_t = sample['input_t']

        if(1):
            a,b = np.random.uniform(0.2,0.6),np.random.uniform(0.2,0.6)
            if(a>b):
                low = a
                high = b
            else:    
                low = b
                high = a

            rdict['input'] = rescale_intensity(input_data,(-0.1,1.2))
        sample.update(rdict)
        return sample


class ToTensor2D(object):

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        ret_input = input_data.transpose(2,0,1)
        ret_input = torch.from_numpy(ret_input).float()

        rdict['input'] = ret_input.float()

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = gt_data.transpose(2,0,1)
                ret_gt = torch.from_numpy(ret_gt).float()

                rdict['gt'] = ret_gt

        sample.update(rdict)
        return sample

    


class RandomColorJitterslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample):
        rdict = {}

        input_data = sample['input']

        jitter = transforms.ColorJitter(brightness=.2, hue=.2)

        rdict['input'] = jitter(input_data)

        sample.update(rdict)
        return sample


class RandomGaussianBlurslice(object):
    def __init__(self,):
        pass
    def __call__(self,sample):
        rdict = {}

        input_data = sample['input']

        blur = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 3))

        rdict['input'] = blur(input_data)

        sample.update(rdict)
        return sample


class RandomNoise2D(object):
    """ 
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.epsilon = 1e-7
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        rdict = {}
        input_data = sample['input']
        
        if torch.rand(1) < self.p:
            sigma = 0.1 + 0.1*np.random.rand(1)
            rdict['input'] = sigma*np.random.randn(*input_data.shape) + input_data
            rdict['input'] -= rdict['input'].min()
            rdict['input'] /= rdict['input'].max() + self.epsilon
            sample.update(rdict)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomIntensityChanges(object):
    def __init__(self,p=0.5):
        self.p = p
        
    def __call__(self,sample,p=0.5):
        rdict = {}
        input_data = sample['input']

        if(torch.rand(1)<self.p):
            mid = np.random.uniform(0,1)
            a,b = np.random.uniform(0,mid),np.random.uniform(mid,1)

            rdict['input'] = rescale_intensity(input_data,(a,b),(0.0,1.0))        
        return sample


class RandomRotation2D(object):
    def __init__(self, degrees,p=0.5, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']

        angle = self.get_params(self.degrees)

        input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

        gt_data = sample['gt'] if self.labeled else None
        gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None


        # Rotation angle chosen at random and rotation happens only on XY plane for both image and label.

        if(torch.rand(1)<p):
            input_rotated[:, :, 0] = rotate(input_data[:, :, 0], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')


            if self.labeled:
                gt_rotated[:, :,0] = rotate(gt_data[:, :,0], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')
                gt_rotated[:, :,1] = rotate(gt_data[:, :,1], float(angle), reshape=False, order=self.order,
                                                    mode='nearest')
                gt_rotated = (gt_rotated > 0.6).astype(float)
        else:
            input_rotated = input_data
            if(self.labeled):
                gt_rotated = gt_data


        # Update the dictionary with transformed image and labels
        rdict['input'] = input_rotated


        if self.labeled:
            rdict['gt'] = gt_rotated
        sample.update(rdict)
        return sample
    

################################################################## For 3D transformations #####################################################################

class ToTensor3D(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self, labeled=True):
        self.labeled = labeled

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        ret_input = input_data.transpose(3, 0, 1, 2)  # Pytorch supports N x C x X_dim x Y_dim x Z_dim
        ret_input = torch.from_numpy(ret_input.copy()).float()
        rdict['input'] = ret_input

        if('input_clean' in list(sample.keys())):
            input_clean = sample['input_clean']
            ret_input = input_data.transpose(3,0,1,2)
            ret_input = torch.from_numpy(ret_input.copy()).float()
            rdict['input_clean'] = ret_input
            

        if self.labeled:
            gt_data = sample['gt']
            if gt_data is not None:
                ret_gt = gt_data.transpose(3, 0, 1, 2)  # Pytorch supports N x C x X_dim x Y_dim x Z_dim
                ret_gt = torch.from_numpy(ret_gt.copy()).float()

                rdict['gt'] = ret_gt
        sample.update(rdict)
        return sample

class ToTensor3D_Recon(object):
    """Convert a PIL image or numpy array to a PyTorch tensor."""

    def __init__(self):
        pass
    def __call__(self, sample):
        # input_data = sample.numpy()
        # ret_input = input_data.transpose(0, 4, 1, 2, 3)
        # ret_input = torch.from_numpy(ret_input)  # Pytorch supports B x N x C x X_dim x Y_dim x Z_dim
        ret_input = torch.permute(sample,(0, 4, 1, 2, 3))
        return ret_input

class RandomRotation3D(object):
    """Make a rotation of the volume's values.
    :param degrees: Maximum rotation's degrees.
    """

    def __init__(self, degrees, p=0.5, axis=0, labeled=True, segment=True):
        self.degrees = degrees
        self.labeled = labeled
        self.segment = segment
        self.p = 0.5
        self.order = 0 if self.segment == True else 5

    @staticmethod
    def get_params(degrees):  # Get random theta value for rotation
        angle = np.random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, sample):
        rdict = {}
        input_data = sample['input']
        if len(sample['input'].shape) != 4:  # C x X_dim x Y_dim x Z_dim
            raise ValueError("Input of RandomRotation3D should be a 4 dimensionnal tensor.")

        if(torch.rand(1)<self.p):
            angle = self.get_params(self.degrees)

            input_rotated = np.zeros(input_data.shape, dtype=input_data.dtype)

            gt_data = sample['gt'] if self.labeled else None
            gt_rotated = np.zeros(gt_data.shape, dtype=gt_data.dtype) if self.labeled else None

            input_rotated = rotate(input_data, float(angle), reshape=False, order=1,mode='nearest')
            if self.labeled:
                gt_rotated = rotate(gt_data, float(angle), reshape=False, order=self.order,mode='nearest')
                gt_rotated = (gt_rotated > 0.5).astype(np.single)
            
            # Update the dictionary with transformed image and labels
            rdict['input'] = input_rotated

            if self.labeled:
                rdict['gt'] = gt_rotated
            sample.update(rdict)
        return sample

class RandomHorizontalFlip3D(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5,labelled=True):
        super().__init__()
        self.p = p
        self.labelled = labelled
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        rdict = {}
        input_data = sample['input']
        gt_data = sample['gt'] if self.labelled else None
        if torch.rand(1) < self.p:
            rdict['input'] = np.flip(input_data,axis=3)
            if(self.labelled):
                rdict['gt'] = np.flip(gt_data,axis=3)
            sample.update(rdict)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class RandomBrightness3D(torch.nn.Module):
    """ 
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.epsilon = 1e-7
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        rdict = {}
        input_data = sample['input']
        
        if torch.rand(1) < self.p:
            rdict['input'] = _blend(input_data,np.zeros_like(input_data),0.8+0.3*np.random.rand(1))
            rdict['input'] -= rdict['input'].min()
            rdict['input'] /= rdict['input'].max() + self.epsilon
            sample.update(rdict)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"



class RandomNoise3D(torch.nn.Module):
    """ 
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.epsilon = 1e-7
    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        rdict = {}
        input_data = sample['input']
        
        if torch.rand(1) < self.p:
            sigma = 0.1 + 0.1*np.random.rand(1)
            rdict['input'] = sigma*np.random.randn(*input_data.shape) + input_data
            rdict['input'] -= rdict['input'].min()
            rdict['input'] /= rdict['input'].max() + self.epsilon
            sample.update(rdict)
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"




from torch import Tensor
def _blend(img1, img2, ratio):
    ratio = float(ratio)
    return np.clip(ratio * img1 + (1.0 - ratio) * img2,0,1)

import torch
import random
import numbers
import numpy as np
from skimage.transform import resize
import torchvision.transforms.functional as F

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 3:
        return img.shape[-3:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def crop(img, h, w, d, height, width, depth):
    return img[:,h:h+height,w:w+width,d:d+depth]

def center_crop(img, output_size):
    """Crop the given PIL Image and resize it to desired size.

        Args:
            img (PIL Image): Image to be cropped. (0,0) denotes the top left corner of the image.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            PIL Image: Cropped image.
        """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    image_width, image_height = img.size
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))
    return crop(img, crop_top, crop_left, crop_height, crop_width)

class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
 
        h, w, d = _get_image_size(img)
        th, tw, td = output_size
        if w == tw and h == th and d==td:
            return 0, 0, h, w, d

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        k = random.randint(0, d - td)
        return i, j, k, th, tw, td

    def __call__(self, img):

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0, 0), self.fill, self.padding_mode)
        
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1], 0), self.fill, self.padding_mode)
        
        # pad the height if needed
        if self.pad_if_needed and img.size[2] < self.size[2]:
            img = F.pad(img, (0, self.size[0] - img.size[1], 0), self.fill, self.padding_mode)


        i, j, k, h, w, d = self.get_params(img, self.size)

        return crop(img, i, j, k, h, w, d)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)