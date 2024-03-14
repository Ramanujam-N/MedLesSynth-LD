import skimage
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import skimage.morphology
import skimage.transform as skiform
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage

class SphereGeneration(Dataset):
    def __init__(self, path, mask_path=None, transform=None):
        super().__init__()
        self.paths = path
        self.mask_path = mask_path
        self.transform = transform
        self.size =(128,128,128)

    def sphere(self, centroid, size = 32, radius = 10):
        xx, yy, zz = np.mgrid[-size:size, -size:size, -size:size]
        circle = (xx - centroid[0] + size) ** 2 + (yy - centroid[1] + size) ** 2 + (zz - centroid[2] + size) ** 2 - radius**2
        mask = (circle < 0)
        return mask

    def lesion_simulation(self,image,brain_mask_img,num_les=3):

        roi = skimage.morphology.binary_erosion(brain_mask_img,skimage.morphology.ball(10))*(image>0.15)

        # Generating centroids within the roi generated above
        x_corr,y_corr,z_corr = np.nonzero(roi[:,:,:])
        centroid_list = []
        for d in range(num_les):
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
            centroid_list.append(centroid_main)
        
        # Generating spheres and combining the masks
        mask_total = np.zeros_like(image)
        for i in range(num_les):
            radius = 10
            mask = self.sphere(centroid_list[i], 32,radius)
            mask_total = np.logical_or(mask,mask_total)

        alpha = np.random.uniform(0.5)
        beta = 1-alpha

        image = alpha*image*(1-mask_total) + beta*mask_total
        image -= image.min()
        image /= image.max()
    
        return image,mask_total
        

    def __getitem__(self, index):

        # Reading the nifty image and brain mask
        nii_img = nib.load(self.paths[index])
        nii_img_affine = nii_img._affine
        nii_img_data = nii_img.get_fdata()
        image,img_crop_para = self.tight_crop_data(nii_img_data)
        image = skiform.resize(image, self.size, order=1, preserve_range=True)
        image -= image.min()
        image /= image.max() + 1e-7

        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)

        else:
            # In case we don't have brain mask then we can close out the holes in the mask aka ventricles
            brain_mask_img = ndimage.binary_fill_holes(image>0,structure=np.ones((3,3,3)))  

        # Random number of lesions generated 
        number_les = 1
        image,label = self.lesion_simulation(image,brain_mask_img,number_les)

        
        return image.astype(np.single),label.astype(np.single),nii_img_affine

    def __len__(self):
        return len(self.paths)

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
