import glob
import torch
from torch.utils.data import Dataset
import nibabel as nib
import skimage.transform as skiform
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.io import imread
import skimage
from tqdm import tqdm
from skimage.exposure import rescale_intensity,equalize_hist
import numbers

def normalize(image):
    image-=image.min()
    image/=image.max() + 1e-7
    return image

class CutPaste2D(Dataset):
    def __init__(self,paths,gt_path = None,image_size =(512,512),type_of_imgs = 'nifty',no_crop=False, transform=None,data='wmh',resize=True, return_size = False,return_orig=False,scale_factor=5):
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

    def read_image(self,index):
        image = skimage.io.imread(self.paths[index],as_gray=True)
        image = skiform.resize(image, self.image_size, order=1, preserve_range=True)
        image = normalize(image)
        if(self.gt_path!=None):
            gt_img = skimage.io.imread(self.gt_path[index],as_gray=True)
            gt_mask = skiform.resize(gt_img, self.image_size, order=0, preserve_range=True)
            gt_mask = gt_mask>0
            roi_mask = None
        else:
            gt_mask = np.zeros_like(image)
            roi_mask = None
        return image,gt_mask
    
    def __getitem__(self, index):

        image,gt_mask = self.read_image(index)

        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,inter_gt = self.read_image(interpolation_choice)


        _mask_img = np.ones(image.shape)

        # Random number of lesions generated 
        image_mask = (image>0.1)*(1-gt_mask)
        x_corr,y_corr = np.nonzero(image_mask)
        random_coord_index = np.random.choice(len(x_corr),1)
        centroid = np.array([x_corr[random_coord_index],y_corr[random_coord_index]])
        width = np.random.uniform(0.1*self.image_size[0],0.4*self.image_size[0])
        alpha = np.random.uniform(0.1,0.9)

        labels = skimage.measure.label(inter_gt,background=0)
        regions = skimage.measure.regionprops(labels)

        reg = np.random.choice(regions)
        (min_row,min_col,max_row,max_col) = reg.bbox
        # print((min_row,min_col,max_row,max_col))
        x_width = max_row - min_row
        y_width = max_col - min_col
        centroid = [int(centroid[0]),int(centroid[1])]
        
        # print(centroid)
        if(centroid[0]-x_width//2 <0):
            centroid[0] += - (centroid[0]-x_width//2)
        if(centroid[0]+np.ceil(x_width/2) >=512):
            centroid[0] -= centroid[0]+np.ceil(x_width/2) - 512
        
        if(centroid[1]-y_width//2 <0):
            centroid[1] += - (centroid[1]-y_width//2)
        if(centroid[1]+np.ceil(y_width/2) >=512):
            centroid[1] -= centroid[1]+np.ceil(y_width/2) - 512
        # print(centroid)

        coords_x = (max(int(centroid[0]-(x_width//2)),0),min(int(centroid[0]+np.ceil(x_width/2)),512)) 
        coords_y = (max(int(centroid[1]-(y_width//2)),0),min(int(centroid[1]+np.ceil(y_width/2)),512))

        gt = gt_mask.copy()
        gt[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1]] = 2


        image[coords_x[0]:coords_x[1],coords_y[0]:coords_y[1]] = inter_image[min_row:max_row,min_col:max_col]

        # plt.subplot(1,4,1)
        # plt.imshow(gt_mask>0)
        # plt.subplot(1,4,2)
        # plt.imshow(inter_gt>0)
        # plt.subplot(1,4,3)
        # plt.imshow(gt>0)
        # plt.subplot(1,4,4)
        # plt.imshow(image)
        # plt.show()
        gt = gt>0
        
        return image.astype(np.single),gt.astype(np.single)
    
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
    
