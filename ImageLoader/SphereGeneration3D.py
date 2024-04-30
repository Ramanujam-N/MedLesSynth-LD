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

class SphereGeneration3D(Dataset):
    def __init__(self, path, mask_path=None,gt_path=None,num_lesions=2, transform=None):
        super().__init__()
        self.paths = path
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.num_lesions = num_lesions
        self.transform = transform
        self.size =(128,128,128)

    def sphere(self, centroid, size = 32, radius = 10):
        xx, yy, zz = np.mgrid[-size:size, -size:size, -size:size]
        circle = (xx - centroid[0] + size) ** 2 + (yy - centroid[1] + size) ** 2 + (zz - centroid[2] + size) ** 2 - radius**2
        mask = (circle < 0)
        return mask

    def simulation(self,image,brain_mask_img,gt_mask,num_les=3):

        roi = skimage.morphology.binary_erosion(brain_mask_img,skimage.morphology.ball(10))*(image>0.15)
        roi = roi*(1-gt_mask)
        output_image = image
        output_mask = gt_mask
        # Generating centroids within the roi generated above
        x_corr,y_corr,z_corr = np.nonzero(roi[:,:,:])
        centroid_list = []
        for d in range(num_les):
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
            centroid_list.append(centroid_main)
        
        # Generating spheres and combining the masks
        # mask_total = np.zeros_like(image)
        for i in range(num_les):
            radius = 10
            mask = self.sphere(centroid_list[i], 64,radius)
            output_mask = np.logical_or(mask,output_mask)

        alpha = np.random.uniform(0.5)
        beta = 1-alpha

        image = alpha*image*(1-output_mask) + beta*output_mask*(1-gt_mask) +alpha*image*gt_mask
        image -= image.min()
        image /= image.max()
    
        return image,output_mask
        
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
        image = skiform.resize(image, self.size, order=1, preserve_range=True ,anti_aliasing=False)
        image -= image.min()
        image /= image.max() + 1e-7

        if(self.gt_path!=None):
            gt_img = nib.load(self.gt_path[index]).get_fdata()
            gt_img = gt_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            gt_mask = skiform.resize(gt_img, self.size, order=0, preserve_range=True)
            gt_mask = gt_mask>0
            roi_mask = None
        elif(self.roi_path!=None):
            roi_img = nib.load(self.roi_path[index]).get_fdata()
            roi_img = roi_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            roi_mask = skiform.resize(roi_img, self.size, order=0, preserve_range=True)
            roi_mask = (roi_mask>1)*(roi_mask<2)
            roi_mask = ndimage.binary_dilation(roi_mask)

            gt_mask = np.zeros_like(image)
        else:
            gt_mask = np.zeros_like(image)
            roi_mask = None

        # if(inter_image and self.which_data!='lits'):
        #     angle = np.random.uniform(0, 180)
        #     axes = np.random.choice([0,1,2],2,replace=False)
        #     input_rotated1 = ndimage.rotate(image, float(angle), axes=axes, reshape=False, mode='nearest')
        #     image = input_rotated1
        
        return image,nii_img_affine,img_crop_para,gt_mask,roi_mask
    
    def __getitem__(self, index):
        image,nii_img_affine,img_crop_para,gt_mask,roi_mask = self.read_image(index)
        clean_image = image

        # interpolation_choice = np.random.choice(len(self.paths))
        # inter_image,nii_img_affine,_,_,_ = self.read_image(interpolation_choice,inter_image=True)
        # clean_inter_image = inter_image


        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)
            
        else:
            brain_mask_img = ndimage.binary_fill_holes(image>0,structure=np.ones((3,3,3)))
        
        param_dict = {}

        number_les = np.random.choice(range(1,self.num_lesions))
        image, label = self.simulation(image, brain_mask_img,gt_mask=gt_mask,num_les=number_les)

        
        # plt.imshow(image[:,:,60])
        # plt.show()
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
