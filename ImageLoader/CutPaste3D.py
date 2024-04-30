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
import skimage
class CutPaste3D(Dataset):
    def __init__(self,paths,gt_path=None,image_size =(128,128,128),type_of_imgs = 'nifty',no_crop=False, transform=None,data='wmh',resize=True, return_size = False,return_orig=False,scale_factor=5):
        self.paths = [*paths]*scale_factor
        self.gt_path = [*gt_path]*scale_factor
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

    def read_image(self,index,inter_image=False):
        nii_img = nib.load(self.paths[index])            
        nii_img_affine = nii_img._affine
        nii_img = nii_img.get_fdata()
        sub_min = 0
        if(nii_img.min()<0):
            sub_min = nii_img.min()
            nii_img = nii_img - nii_img.min()
        image,img_crop_para = self.tight_crop_data(nii_img)
        image+=sub_min
        image = skiform.resize(image, self.image_size, order=1, preserve_range=True ,anti_aliasing=False)
        image -= image.min()
        image /= image.max() + 1e-7

        if(self.gt_path!=None):
            gt_img = nib.load(self.gt_path[index]).get_fdata()
            gt_img = gt_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            gt_mask = skiform.resize(gt_img, self.image_size, order=0, preserve_range=True)
            gt_mask = gt_mask>0
            roi_mask = None
        else:
            gt_mask = np.zeros_like(image)
            roi_mask = None
        return image,nii_img_affine,img_crop_para,gt_mask,roi_mask
    
    def __getitem__(self,index):
        image,affine,img_crop_para,gt_mask,roi_mask = self.read_image(index)
        clean_image = image
        shape = image.shape
        
        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,nii_img_affine,_,inter_gt,_ = self.read_image(interpolation_choice,inter_image=True)
        clean_inter_image = inter_image

        data_dict = {}

        image_mask = (image>0.1)*(1-gt_mask)
        outline_mask = (image>0.1)
        x_corr,y_corr,z_corr = np.nonzero(image_mask)
        random_coord_index = np.random.choice(len(x_corr),1)
        centroid = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
        width = np.random.uniform(0.1*self.image_size[0],0.4*self.image_size[0])
        alpha = np.random.uniform(0.1,0.9)

        # centroid = (int(centroid[0]),int(centroid[1]),int(centroid[2]))
        # coords_x = (max(int(centroid[0]-width),0),min(int(centroid[0]+width),128)) 
        # coords_y = (max(int(centroid[1]-width),0),min(int(centroid[1]+width),128))
        # coords_z = (max(int(centroid[2]-width),0),min(int(centroid[2]+width),128))

        # gt = np.zeros_like(image)
        # gt[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1],coords_z[0]:coords_z[1]] = 1
        # gt*=image_mask


        # image[gt>0] = inter_image[gt>0] 

        labels = skimage.measure.label(inter_gt,background=0)
        regions = skimage.measure.regionprops(labels)

        reg = np.random.choice(regions)
        (min_row,min_col,min_dep,max_row,max_col,max_dep) = reg.bbox
        # print((min_row,min_col,max_row,max_col))
        x_width = max_row - min_row
        y_width = max_col - min_col
        z_width = max_dep - min_dep
        centroid = [int(centroid[0]),int(centroid[1]),int(centroid[2])]
        
        # print(centroid)
        if(centroid[0]-x_width//2 <0):
            centroid[0] += - (centroid[0]-x_width//2)
        if(centroid[0]+np.ceil(x_width/2) >=128):
            centroid[0] -= centroid[0]+np.ceil(x_width/2) - 128
        
        if(centroid[1]-y_width//2 <0):
            centroid[1] += - (centroid[1]-y_width//2)
        if(centroid[1]+np.ceil(y_width/2) >=128):
            centroid[1] -= centroid[1]+np.ceil(y_width/2) - 128
        
        if(centroid[2]-z_width//2 <0):
            centroid[2] += - (centroid[2]-z_width//2)
        if(centroid[2]+np.ceil(z_width/2) >=128):
            centroid[2] -= centroid[2]+np.ceil(z_width/2) - 128
        # print(centroid)

        coords_x = (max(int(centroid[0]-(x_width//2)),0),min(int(centroid[0]+np.ceil(x_width/2)),128)) 
        coords_y = (max(int(centroid[1]-(y_width//2)),0),min(int(centroid[1]+np.ceil(y_width/2)),128))
        coords_z = (max(int(centroid[2]-(z_width//2)),0),min(int(centroid[2]+np.ceil(z_width/2)),128))

        gt = gt_mask.copy()
        gt[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1],coords_z[0]:coords_z[1]] = 2


        image[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1],coords_z[0]:coords_z[1]] = inter_image[min_row:max_row,min_col:max_col,min_dep:max_dep]

        image = image*outline_mask
        gt = gt>0
        gt = gt*outline_mask

        # image = np.expand_dims(image,-1).astype(np.single)
        # gt = np.expand_dims(gt>0,-1).astype(np.single)

        # data_dict['input'] = image
        # data_dict['gt'] = gt

        # if(self.return_orig):
        #     data_dict['affine'] = affine
        #     data_dict['shape'] = shape
        #     data_dict['crop_para'] = img_crop_para
        #     data_dict['path'] = self.paths[index]


        # if(self.transform):
        #     data_dict = self.transform(data_dict)

        # return data_dict
        return image.astype(np.single),gt.astype(np.single),nii_img_affine


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
    
