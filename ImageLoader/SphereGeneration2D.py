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

def normalize(image):
    image-=image.min()
    image/=image.max() + 1e-7
    return image


class SphereGeneration2D(Dataset):
    def __init__(self, path, mask_path=None, gt_path=None,num_lesions=2, transform=None):
        super().__init__()
        self.paths = path
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.num_lesions = num_lesions
        self.transform = transform
        self.size =(512,512)

    def sphere(self, centroid, size = 32, radius = 10):
        xx, yy = np.mgrid[-size:size, -size:size]
        circle = (xx - centroid[0] + size) ** 2 + (yy - centroid[1] + size) ** 2 - radius**2
        mask = (circle < 0)
        return mask

    def lesion_simulation(self,image,brain_mask_img,gt_mask,num_les=3):

        # roi = skimage.morphology.binary_erosion(brain_mask_img,skimage.morphology.cir(10))*(image>0.15)
        roi = brain_mask_img*(1-gt_mask)
        output_image = image
        output_mask = gt_mask
        # Generating centroids within the roi generated above
        x_corr,y_corr = np.nonzero(roi[:,:])
        centroid_list = []
        for d in range(num_les):
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index]])
            centroid_list.append(centroid_main)
        
        # Generating spheres and combining the masks
        
        for i in range(num_les):
            radius = 100
            mask = self.sphere(centroid_list[i], 256,radius)
            output_mask = np.logical_or(mask,output_mask)

        alpha = np.random.uniform(0.8,0.9)
        beta = 1-alpha

        image = alpha*image*(1-output_mask) + beta*output_mask*(1-gt_mask) +alpha*image*gt_mask
        image -= image.min()
        image /= image.max()
    
        return image,output_mask

    def read_image(self,index):
        image = skimage.io.imread(self.paths[index],as_gray=True)
        image = skiform.resize(image, self.size, order=1, preserve_range=True)
        image = normalize(image)
        if(self.gt_path!=None):
            gt_img = skimage.io.imread(self.gt_path[index],as_gray=True)
            gt_mask = skiform.resize(gt_img, self.size, order=0, preserve_range=True)
            gt_mask = gt_mask>0
            roi_mask = None
        else:
            gt_mask = np.zeros_like(image)
            roi_mask = None
        return image,gt_mask
    def __getitem__(self, index):

        image,gt_mask = self.read_image(index)

        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,_ = self.read_image(interpolation_choice)


        _mask_img = np.ones(image.shape)

        # Random number of lesions generated 
        number_les = np.random.choice(range(1,self.num_lesions))
        image,label = self.lesion_simulation(image,_mask_img,gt_mask,number_les)

        
        return image.astype(np.single),label.astype(np.single)

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
