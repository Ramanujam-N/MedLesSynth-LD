import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
from skimage.exposure import rescale_intensity,equalize_hist
import numbers

class ImageLoader3D(Dataset):
    def __init__(self,paths,gt_paths,paths_clean=None,image_size =(128,128,128),type_of_imgs = 'nifty',no_crop=False, transform=None,data='wmh',resize=True, return_size = False,return_orig=False):
        self.paths = paths
        self.gt_paths = gt_paths
        self.paths_clean = paths_clean
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.no_crop = no_crop
        self.resize = resize
        self.return_size = return_size
        self.return_orig = return_orig
        self.data = data
        if isinstance(image_size, numbers.Number):
            self.image_size = (int(image_size), int(image_size))
    def __len__(self,):
        return len(self.paths)

    def __getitem__(self,index):


        image = nib.load(self.paths[index])
        affine = image._affine
        pixdim = image.header['pixdim']
        image = image.get_fdata()
        shape = image.shape
        img_crop_para = []
        gt = nib.load(self.gt_paths[index]).get_fdata()
        orig_gt = gt

        data_dict = {}

        if(not self.no_crop):
            image,img_crop_para = self.tight_crop_data(image)
            shape = image.shape
            gt = gt[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
        
        if(self.resize):
            image = skiform.resize(image, self.image_size, order = 1, preserve_range=True )
            gt = skiform.resize(gt, self.image_size, order = 0, preserve_range=True)

        if(np.isnan(image).sum() or np.isnan(gt).sum()):
            print('Nan image:',self.paths[index])

        image-=image.min()
        image/=image.max()+1e-12

        image = np.expand_dims(image,-1).astype(np.single)
        gt = np.expand_dims(gt>0,-1).astype(np.single)

        data_dict['input'] = image
        data_dict['gt'] = gt

        if(self.return_orig):
            data_dict['affine'] = affine
            data_dict['orig'] = orig_gt
            data_dict['shape'] = shape
            data_dict['pixdim'] = pixdim
            data_dict['crop_para'] = img_crop_para
            data_dict['path'] = self.paths[index]

        if(self.transform):
            data_dict = self.transform(data_dict)

        return data_dict
    
    def cut_zeros1d(self, im_array):
        '''
     Find the window for cropping the data closer to the brain
     :param im_array: input array
     :return: starting and end indices, and length of non-zero intensity values
        '''

        im_list = list(im_array > 0)
        start_index = im_list.index(1)
        end_index = im_list[::-1].index(1)
        length = len(im_array[start_index:]) - end_index
        return start_index, end_index, length

    def tight_crop_data(self, img_data):
        '''
     Crop the data tighter to the brain
     :param img_data: input array
     :return: cropped image and the bounding box coordinates and dimensions.
        '''

        row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
        col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
        stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], [rsid, rlen, csid, clen, ssid, slen]
    
