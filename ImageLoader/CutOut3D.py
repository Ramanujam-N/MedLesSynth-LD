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

class CutOut3D(Dataset):
    def __init__(self,paths,paths_clean=None,image_size =(128,128,128),type_of_imgs = 'nifty',no_crop=False, transform=None,data='wmh',resize=True, return_size = False,return_orig=False):
        self.paths = paths
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
        inter_index = np.random.choice(len(self.paths),1).item()
        inter_image = nib.load(self.paths[inter_index]).get_fdata()

        image = nib.load(self.paths[index])
        affine = image._affine
        pixdim = image.header['pixdim']
        image = image.get_fdata()
        shape = image.shape
        img_crop_para = []

        data_dict = {}

        image,img_crop_para = self.tight_crop_data(image)
        shape = image.shape
        inter_image,inter_img_crop_para = self.tight_crop_data(inter_image)
        
        
        image = skiform.resize(image, self.image_size, order = 1, preserve_range=True )
        inter_image = skiform.resize(inter_image, self.image_size, order = 1, preserve_range=True )


        image-=image.min()
        image/=image.max()+1e-12

        inter_image-=inter_image.min()
        inter_image/=inter_image.max()+1e-12

        image_mask = image>0.1
        x_corr,y_corr,z_corr = np.nonzero(image_mask)
        random_coord_index = np.random.choice(len(x_corr),1)
        centroid = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
        width = np.random.uniform(0.1*self.image_size[0],0.4*self.image_size[0])
        alpha = np.random.uniform(0.1,0.9)

        centroid = (int(centroid[0]),int(centroid[1]),int(centroid[2]))
        coords_x = (max(int(centroid[0]-width),0),min(int(centroid[0]+width),128)) 
        coords_y = (max(int(centroid[1]-width),0),min(int(centroid[1]+width),128))
        coords_z = (max(int(centroid[2]-width),0),min(int(centroid[2]+width),128))

        gt = np.zeros_like(image)
        gt[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1],coords_z[0]:coords_z[1]] = 1
        gt*=image_mask


        image[gt>0] = inter_image[gt>0] 
        
        image = np.expand_dims(image,-1).astype(np.single)
        gt = np.expand_dims(gt>0,-1).astype(np.single)

        data_dict['input'] = image
        data_dict['gt'] = gt

        if(self.return_orig):
            data_dict['affine'] = affine
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
    
