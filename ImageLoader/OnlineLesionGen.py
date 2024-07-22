import skimage
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import skimage.morphology
import skimage.transform as skiform
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy

# import plotly.graph_objects as go
import cupy as cp
from cupyx.scipy.ndimage import binary_closing, binary_erosion, binary_opening, rotate, binary_dilation, binary_fill_holes
from cupyx.scipy.ndimage import gaussian_filter as gaussian_filter_gpu

import cupy as cp
import cucim
#################################

class OnlineLesionGen(Dataset):
    def __init__(self, img_path, gt_path=None, mask_path = None, roi_path=None, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15,rician=False):
        self.paths = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.size = size
        self.have_noise = have_noise
        self.have_smoothing = have_smoothing
        self.have_small = have_small
        self.have_edema = have_edema
        self.img_type = type_of_imgs
        self.return_param = return_param
        self.dark = dark
        self.which_data = which_data

        self.paths = img_path
        self.gt_path = gt_path
        self.mask_path = mask_path
        self.roi_path = roi_path
        self.transform = transform
        self.size = size
        self.have_noise = have_noise
        self.have_smoothing = have_smoothing
        self.have_small = have_small
        self.have_edema = have_edema
        self.img_type = type_of_imgs
        self.return_param = return_param
        self.dark = dark
        self.which_data = which_data
        self.perturb = perturb
        self.ranges = semi_axis_range
        self.num_lesions = num_lesions
        self.centroid_scaling = centroid_scale
        self.range_sampling = range_sampling
        self.num_ellipses = num_ellipses   
        


    def ellipsoid(self, coord=(1,2,1), semi_a=4, semi_b=34, semi_c=34, alpha=cp.pi/4, beta=cp.pi/4, gamma=cp.pi/4, img_dim=128):
        
        # coord = tuple(cp.asnumpy(coord))
        # semi_a = float(cp.asnumpy(semi_a))
        # semi_b = float(cp.asnumpy(semi_b))
        # semi_c = float(cp.asnumpy(semi_c))
        # img_dim = int(cp.asnumpy(img_dim))
        
        # Create coordinate grids
        x = cp.linspace(-img_dim, img_dim, img_dim*2)
        y = cp.linspace(-img_dim, img_dim, img_dim*2)
        z = cp.linspace(-img_dim, img_dim, img_dim*2)
        x, y, z = cp.meshgrid(x, y, z)

        # Take the centering into effect
        x = (x - coord[0] + img_dim)
        y = (y - coord[1] + img_dim)
        z = (z - coord[2] + img_dim)

        ellipsoid_std_axes = cp.stack([x, y, z], 0)

        # Negate rotation angles
        alpha = -alpha.get()
        beta = -beta.get()
        gamma = -gamma.get()

        # Create rotation matrices

        rotation_x = np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])
        rotation_x = cp.array(rotation_x)

        rotation_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                                [0, 1, 0],
                                [-np.sin(beta), 0, np.cos(beta)]])
        rotation_y = cp.array(rotation_y)

        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])
        rotation_z = cp.array(rotation_z)

        # Compute full rotation matrix
        rot_matrix = rotation_x @ rotation_y @ rotation_z

        # Apply rotation
        ellipsoid_rot_axes = cp.tensordot(rot_matrix, ellipsoid_std_axes, axes=([1,0]))
        x, y, z = ellipsoid_rot_axes

        # Square coordinates
        x **= 2
        y **= 2
        z **= 2

        # Compute semi-axis squares
        a = semi_a**2
        b = semi_b**2
        c = semi_c**2

        # Create ellipsoid
        ellipsoid = x/a + y/b + z/c - 1
        ellipsoid = ellipsoid < 0

        return ellipsoid

    def gaussian_small_shapes(self,image_mask,small_sigma = [1,1,1]):
        selem_gpu = cp.ascontiguousarray(cp.ones((15, 15, 15), dtype=bool))
        image_mask = binary_erosion(image_mask,selem_gpu)
        index = -1
        while(index==-1):
            noise = cp.random.normal(size = image_mask.shape)
            smoothed_noise = gaussian_filter_gpu(noise,sigma=small_sigma) #+ gaussian_filter(noise,sigma=[4,4,4]) + gaussian_filter(noise,sigma=[3,3,3]) + gaussian_filter(noise,sigma=[2,2,2]) + 0.1*gaussian_filter(noise,sigma=[1,1,1])
            smoothed_noise -= smoothed_noise.min()
            smoothed_noise /= smoothed_noise.max()

            bg_mask = (smoothed_noise>0.3)*(image_mask)
            mask = (1-bg_mask)*(image_mask)

            labelled = cucim.skimage.measure.label(mask*2,background = 0)         
            regions = cucim.skimage.measure.regionprops(labelled)

            total_brain_area = np.sum(image_mask)
            count = 0
            old_area = 0 

            for region in regions:
                if(region.area<total_brain_area*0.01 and region.area>total_brain_area*(0.0001)):
                    if(region.area >old_area):
                        old_area = region.area
                        index = count
                count+=1

            if(index!=-1):
                label = regions[index].label

                label_sim = labelled == label

                return label_sim

    def gaussian_noise(self, sigma=1.0, size=256, range_min=-0.3, range_max=1.0):
        # Generate random noise on GPU
        noise = cp.random.random((size, size, size))

        # Apply Gaussian filters
        gaussian_noise = gaussian_filter_gpu(noise, sigma) + \
                         0.5 * gaussian_filter_gpu(noise, sigma/2) + \
                         0.25 * gaussian_filter_gpu(noise, sigma/4)

        # Compute min and max
        gaussian_noise_min = cp.min(gaussian_noise)
        gaussian_noise_max = cp.max(gaussian_noise)

        # Normalize the noise
        tex_noise = ((gaussian_noise - gaussian_noise_min) * (range_max - range_min) / 
                     (gaussian_noise_max - gaussian_noise_min)) + range_min

        return tex_noise
    
    def shape_generation(self, scale_centroids, centroid_main, num_ellipses, semi_axes_range, perturb_sigma=[1,1,1], image_mask=1):
        # Convert inputs to GPU
        centroid_main_gpu = cp.asarray(centroid_main)
        image_mask_gpu = cp.asarray(image_mask)

        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main_gpu.T + (cp.random.random((num_ellipses, 3)) * scale_centroids)

        # Selection of the semi axis length
        random_major_axes = cp.random.uniform(semi_axes_range[0], semi_axes_range[1], (num_ellipses, 1))
        random_minor_axes = cp.concatenate([cp.ones((num_ellipses, 1)), cp.random.uniform(0.1, 0.8, size=(num_ellipses, 2))], 1)
        random_semi_axes = random_major_axes * random_minor_axes

        # Permuting the axes
        random_semi_axes = cp.random.permutation(random_semi_axes.T).T

        # Random rotation angles for the ellipsoids
        random_rot_angles = cp.random.uniform(size=(num_ellipses, 3)) * cp.pi

        out_cpu = []
        for i in range(num_ellipses):
            out_cpu.append(self.ellipsoid(random_centroids[i], *random_semi_axes[i], *random_rot_angles[i], img_dim=128).get())

        # out = cp.logical_or.reduce(out) * image_mask_gpu
        # print(type(out), type(image_mask_gpu))
        # out = cp.any(out, axis=0) * image_mask_gpu


        # Regions analysis (this part might need to be done on CPU)
        # out_cpu = cp.asnumpy(out)

        out_cpu = np.logical_or.reduce(out_cpu)*image_mask_gpu.get()
        out_gpu = cp.asarray(out_cpu)

        regions = skimage.measure.regionprops(np.int16(out_cpu * 2))
        if not regions:
            return cp.zeros_like(out_gpu), -1

        bounding_box = regions[0].bbox

        noise_b_box = cp.random.normal(size=(self.size, self.size, self.size))
        noise_b_box = gaussian_filter_gpu(noise_b_box, sigma=perturb_sigma)
        noise_b_box -= noise_b_box.min()
        noise_b_box /= noise_b_box.max()

        thresholded_b_box = cp.zeros_like(out_gpu)
        thresholded_b_box[bounding_box[0]:bounding_box[3],
                          bounding_box[1]:bounding_box[4],
                          bounding_box[2]:bounding_box[5]] = (noise_b_box > 0.6)[bounding_box[0]:bounding_box[3],
                                                                                 bounding_box[1]:bounding_box[4],
                                                                                 bounding_box[2]:bounding_box[5]]

        # Labeling (this part needs to be done on CPU as CuPy doesn't have an equivalent function)
        thresholded_b_box_cpu = cp.asnumpy(thresholded_b_box)
        labelled_threshold_b_box, nlabels = skimage.measure.label(thresholded_b_box_cpu, return_num=True)
        labelled_threshold_b_box = cp.asarray(labelled_threshold_b_box)

        labels_in_big_lesion = cp.unique(out_gpu * labelled_threshold_b_box)
        labels_tob_removed = cp.setdiff1d(cp.arange(1, nlabels+1), labels_in_big_lesion)

        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out_gpu + labelled_threshold_b_box > 0

        return final_region, 1

    def simulation(self, image, inter_image, image_mask, num_lesions,gt_mask,roi_mask):
        param_dict = {}
        image_mask = np.array(image_mask).astype(np.float32)
        # Highlight: Converted to CuPy
        image_gpu = cp.asarray(image)
        inter_image_gpu = cp.asarray(inter_image)
        image_mask_gpu = cp.asarray(image_mask).astype(cp.float32)

        selem_gpu = cp.ascontiguousarray(cp.ones((15, 15, 15), dtype=bool))

        outer_roi_gpu = binary_closing(image_mask_gpu, structure=selem_gpu)
        outer_roi_gpu = binary_erosion(outer_roi_gpu, structure=selem_gpu)
        outer_roi_gpu = binary_erosion(outer_roi_gpu, structure=selem_gpu)
        inner_roi_gpu = binary_opening(image_mask_gpu, structure=selem_gpu)

        roi_gpu = cp.logical_and(outer_roi_gpu, inner_roi_gpu).astype(cp.float32)

        roi_with_masks_gpu = roi_gpu
        output_image_gpu = image_gpu
        output_mask_gpu = cp.zeros_like(image_mask_gpu)

        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']

        if self.which_data == 'valdo':
            num_lesions = cp.random.randint(5, 20)
            num_lesions = int(cp.asnumpy(num_lesions))
            ranges = [(1, 1.5),(1, 1.5)]
            centroid_scaling = 1

        # num_lesions = int(cp.asnumpy(num_lesions))
        num_lesions = int(cp.random.randint(low=num_lesions[0],high=num_lesions[1]))
        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0

            x_corr, y_corr, z_corr = cp.nonzero(roi_with_masks_gpu * image_mask_gpu)
            random_coord_index = cp.random.choice(len(x_corr), 1)
            centroid_main = cp.array([x_corr[random_coord_index], y_corr[random_coord_index], z_corr[random_coord_index]])

            scale_centroid = cp.random.uniform(2, self.centroid_scaling)
            num_ellipses = 15
            # if self.which_data == 'all':
            #     semi_axes_range = ranges[int(cp.asnumpy(cp.random.choice(4, size=1, p=cp.array([0.35,0.4,0.15,0.1])))[0])]
            # else:
            #     semi_axes_range = ranges[int(cp.asnumpy(cp.random.choice(2, size=1, p=cp.array([0.7,0.3])))[0])]
            semi_axes_range = self.ranges[int(cp.random.choice(len(self.ranges),1,p=self.range_sampling))]

            if semi_axes_range == (2,5):
                alpha = cp.random.uniform(0.5, 0.8)
                beta = 1 - alpha
            if semi_axes_range != (2,5):
                alpha = cp.random.uniform(0.6, 0.8)
                beta = 1 - alpha

            smoothing_mask = 0.5
            smoothing_image = 0.1

            tex_sigma = cp.random.uniform(0.4, 0.7)
            range_min = cp.random.uniform(0, 0.5)
            range_max = cp.random.uniform(0.7, 1)
            perturb_sigma = cp.random.uniform(0.5, 5)

            if semi_axes_range == (2,5):
                small_sigma = cp.random.uniform(1, 2)
                out = self.gaussian_small_shapes(image_mask_gpu, small_sigma)
            else:   
                out, shape_status = self.shape_generation(scale_centroid, centroid_main, num_ellipses, semi_axes_range, perturb_sigma, image_mask=image_mask_gpu)
                while shape_status == -1:
                    print(shape_status)
                    random_coord_index = cp.random.choice(len(x_corr), 1)
                    centroid_main = cp.array([x_corr[random_coord_index], y_corr[random_coord_index], z_corr[random_coord_index]])
                    out, shape_status = self.shape_generation(scale_centroid, centroid_main, num_ellipses, semi_axes_range, perturb_sigma, image_mask=image_mask_gpu)

            if semi_axes_range not in [(2,5), (3,5)]:
                semi_axes_range_edema = (semi_axes_range[0]+5, semi_axes_range[1]+5)
                if self.which_data == 'valdo':
                    semi_axes_range_edema = (semi_axes_range[0]+1.5, semi_axes_range[1]+1.5)

                tex_sigma_edema = cp.random.uniform(1, 1.5)
                beta = 1 - alpha

                gamma = cp.random.uniform(0.2, 0.5)
                out_edema, shape_status = self.shape_generation(scale_centroid, centroid_main, num_ellipses, semi_axes_range_edema, perturb_sigma, image_mask=image_mask_gpu)
                while shape_status == -1:
                    print(shape_status)
                    random_coord_index = cp.random.choice(len(x_corr), 1)
                    centroid_main = cp.array([x_corr[random_coord_index], y_corr[random_coord_index], z_corr[random_coord_index]])
                    out_edema, shape_status = self.shape_generation(scale_centroid, centroid_main, num_ellipses, semi_axes_range_edema, perturb_sigma, image_mask=image_mask_gpu)

            output_mask_gpu = cp.logical_or(output_mask_gpu, out)

            if self.have_noise and semi_axes_range != (2,5):
                tex_noise = self.gaussian_noise(tex_sigma, self.size, range_min, range_max)
            else:
                tex_noise = 1.0

            if self.have_smoothing and self.have_edema and semi_axes_range not in [(2,5), (3,5)]:
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema, self.size, range_min, range_max)

                smoothed_les = beta * gaussian_filter_gpu(out_edema * (gamma * (1-out) * tex_noise_edema + 4*gamma * out * tex_noise), sigma=smoothing_mask)
                if self.which_data == 'valdo':
                    smoothed_les = beta * gaussian_filter_gpu(out_edema * (3*gamma * (1-out) * tex_noise_edema - 3*gamma * out * tex_noise), sigma=smoothing_mask)

                smoothed_out = gaussian_filter_gpu(0.5 * output_image_gpu + 0.5 * inter_image_gpu, sigma=smoothing_image)

                smoothed_out -= smoothed_out.min()
                smoothed_out /= smoothed_out.max()

                image1 = alpha * output_image_gpu + smoothed_les
                image2 = alpha * smoothed_out + smoothed_les

                image1[out_edema > 0] = image2[out_edema > 0]
                image1[image1 < 0] = 0

                output_image_gpu = image1
                output_image_gpu -= output_image_gpu.min()
                output_image_gpu /= output_image_gpu.max()

            if self.have_smoothing and semi_axes_range in [(2,5), (3,5)]:
                smoothed_les = gaussian_filter_gpu(out * tex_noise, sigma=smoothing_mask)
                smoothed_out = gaussian_filter_gpu(0.5 * output_image_gpu + 0.5 * inter_image_gpu, sigma=smoothing_image)

                if self.dark:
                    image1 = alpha * output_image_gpu - beta * smoothed_les
                    image2 = alpha * smoothed_out - beta * smoothed_les
                else:
                    image1 = alpha * output_image_gpu + beta * smoothed_les
                    image2 = alpha * smoothed_out + beta * smoothed_les
                image1[out > 0] = image2[out > 0]
                image1[image1 < 0] = 0.1

            output_image_gpu = image1
            output_image_gpu -= output_image_gpu.min()
            output_image_gpu /= output_image_gpu.max()

            roi_with_masks_gpu *= (1 - output_mask_gpu) > 0

            total_params = [scale_centroid, num_ellipses, semi_axes_range, alpha, beta, gamma, smoothing_mask,
                            tex_sigma, range_min, range_max, tex_sigma_edema, perturb_sigma]

            for j in range(len(total_params)):
                param_dict[f"{i}_{total_param_list[j]}"] = cp.asnumpy(total_params[j]) if isinstance(total_params[j], cp.ndarray) else total_params[j]

        param_dict['num_lesions'] = num_lesions

        if self.return_param:
            return cp.asnumpy(output_image_gpu), cp.asnumpy(output_mask_gpu), param_dict
        else:
            return cp.asnumpy(output_image_gpu), cp.asnumpy(output_mask_gpu)


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
        #image = skiform.resize(image, self.size, order=1, preserve_range=True ,anti_aliasing=True)
        image -= image.min()
        image /= image.max() + 1e-7
        if(self.gt_path!=None):
            gt_img = nib.load(self.gt_path[index]).get_fdata()
            gt_img = gt_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            #gt_img = skiform.resize(gt_img, self.size, order=0, preserve_range=True)
            gt_mask = gt_img>0
            roi_mask = None
        elif(self.roi_path!=None):
            roi_img = nib.load(self.roi_path[index]).get_fdata()
            roi_img = roi_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1], img_crop_para[2]:img_crop_para[2] + img_crop_para[3], img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            #roi_img = skiform.resize(roi_img, self.size, order=0, preserve_range=True)
            roi_mask = (roi_img>1)*(roi_img<2)
            roi_mask = scipy.ndimage.binary_dilation(roi_mask)

            gt_mask = np.zeros_like(image)
        else:
            gt_mask = np.zeros_like(image)
            roi_mask = None

        if(inter_image and self.which_data!='lits'):
            angle = np.random.uniform(0, 180)
            axes = np.random.choice([0,1,2],2,replace=False)
            input_rotated1 = scipy.ndimage.rotate(image, float(angle), axes=axes, reshape=False, mode='nearest')
            image = input_rotated1
        
        return image,nii_img_affine,img_crop_para,gt_mask,roi_mask

    def __getitem__(self, index):
        image,nii_img_affine,img_crop_para,gt_mask,roi_mask = self.read_image(index)
        clean_image = image
        

        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,nii_img_affine,_,_,_ = self.read_image(interpolation_choice,inter_image=True)
        clean_inter_image = inter_image


        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            # brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)
            
        else:
            brain_mask_img = scipy.ndimage.binary_fill_holes(image>0,structure=np.ones((3,3,3)))
        
        param_dict = {}

        if(self.return_param):
            # image = cp.array(image)
            # inter_image = cp.array(inter_image) 
            # brain_mask_img = cp.array(brain_mask_img)
            # gt_mask = cp.array(gt_mask)
            # roi_mask = cp.array(roi_mask)
            
            image, label, param_dict = self.simulation(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
            # image, label, param_dict = self.simulation_onmask(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
        else:
            image, label = self.simulation(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
            # image, label = self.simulation_onmask(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)

        return image.astype(np.single),label.astype(np.single),nii_img_affine,self.paths[index],param_dict

    def __len__(self):
        """Return the dataset size."""
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
