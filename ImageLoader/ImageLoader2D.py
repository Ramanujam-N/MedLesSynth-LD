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
import torchvision
import numbers


class ImageLoader2D(Dataset):
    def __init__(self,paths,gt_paths,paths_clean=None,image_size =(128,128),type_of_imgs = 'png', recon=False,no_crop=False, transform=None,data='busi',return_size = False,return_orig=False):
        self.paths = paths
        self.gt_paths = gt_paths
        self.paths_clean = paths_clean
        self.transform = transform
        self.image_size = image_size
        self.type_of_imgs = type_of_imgs
        self.return_size = return_size
        self.return_orig = return_orig
        if isinstance(image_size, numbers.Number):
            self.image_size = (int(image_size), int(image_size))
    def __len__(self,):
        return len(self.paths)

    def __getitem__(self,index):

        image = torchvision.io.read_image(self.paths[index],torchvision.io.ImageReadMode.GRAY).float()
        gt = torchvision.io.read_image(self.gt_paths[index],torchvision.io.ImageReadMode.GRAY).float()
        orig_gt = gt
        # print(self.paths,self.gt_paths)
        # plt.subplot(1,2,1)
        # plt.imshow(image[0].cpu().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(gt[0].cpu().numpy())
        # plt.savefig('./images/Real_1_12_23/busi'+str(5654)+'.png')
        # print(self.paths[index],self.gt_paths[index])
        # exit(0)
        image = torchvision.transforms.functional.resize(image, size = self.image_size ,interpolation = torchvision.transforms.InterpolationMode.BILINEAR,antialias=True)
        gt = torchvision.transforms.functional.resize(gt, size = self.image_size,interpolation = torchvision.transforms.InterpolationMode.NEAREST,antialias=True)

        gt = gt > 0

        image-=image.min()
        image/=image.max() + 1e-12

        #image = np.expand_dims(image,-1).astype(np.single)
        
        gt = (gt>0).float()

        data_dict={}
        data_dict['input'] = image
        data_dict['gt'] = gt
        if(self.return_orig):
            data_dict['orig'] = orig_gt
            data_dict['shape'] = orig_gt.shape
            data_dict['crop_para'] = [0,0,0,0]

        # if(self.transform):
        #     data_dict = self.transform(data_dict)
        
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