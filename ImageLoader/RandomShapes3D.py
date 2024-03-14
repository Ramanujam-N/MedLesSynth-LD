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

class RandomShapes3D(Dataset):
    def __init__(self, img_path, mask_path = None, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15):
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
        self.perturb = perturb
        self.ranges = semi_axis_range
        self.num_lesions = num_lesions
        self.centroid_scaling = centroid_scale
        self.range_sampling = range_sampling
        self.num_ellipses = num_ellipses        

    def gaussian_small_shapes(self,image_mask,small_sigma = [1,1,1]):

        image_mask = skimage.morphology.binary_erosion(image_mask,skimage.morphology.cube(5))
        index = -1
        while(index==-1):
            noise = np.random.normal(size = self.size)
            smoothed_noise = gaussian_filter(noise,sigma=small_sigma) #+ gaussian_filter(noise,sigma=[4,4,4]) + gaussian_filter(noise,sigma=[3,3,3]) + gaussian_filter(noise,sigma=[2,2,2]) + 0.1*gaussian_filter(noise,sigma=[1,1,1])
            smoothed_noise -= smoothed_noise.min()
            smoothed_noise /= smoothed_noise.max()

            bg_mask = (smoothed_noise>0.3)*(image_mask)
            mask = (1-bg_mask)*(image_mask)

            labelled = skimage.measure.label(mask*2,background = 0)         
            regions = skimage.measure.regionprops(labelled)

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
        
    def ellipsoid(self,coord=(1,2,1),semi_a = 4, semi_b = 34, semi_c = 34, alpha=np.pi/4, beta=np.pi/4, gamma=np.pi/4,img_dim=(128,128,128)):
        x = np.linspace(-img_dim[0]//2,np.ceil(img_dim[0]//2),img_dim[0])
        y = np.linspace(-img_dim[1]//2,np.ceil(img_dim[1]//2),img_dim[1])
        z = np.linspace(-img_dim[2]//2,np.ceil(img_dim[2]//2),img_dim[2])
        x,y,z = np.meshgrid(x,y,z)

        # Take the centering into effect   
        x=(x - coord[0] + img_dim[0]//2)
        y=(y - coord[1] + img_dim[1]//2)
        z=(z - coord[2] + img_dim[2]//2)

        ellipsoid_std_axes = np.stack([x,y,z],0)

        alpha = -alpha
        beta = -beta
        gamma = -gamma    

        rotation_x = np.array([[1, 0, 0],
                                [0, np.cos(alpha), -np.sin(alpha)],
                                [0, np.sin(alpha), np.cos(alpha)]])
        
        rotation_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                                [0, 1, 0],
                                [-np.sin(beta), 0, np.cos(beta)]])
        
        rotation_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                [np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])

        rot_matrix = rotation_x@rotation_y@rotation_z
        ellipsoid_rot_axes = np.tensordot(rot_matrix,ellipsoid_std_axes,axes=([1,0]))

        x,y,z = ellipsoid_rot_axes
        x**=2
        y**=2
        z**=2

        a = semi_a**2
        b = semi_b**2
        c = semi_c**2

        ellipsoid = x/a + y/b + z/c - 1
        ellipsoid = ellipsoid<0 
        return ellipsoid
        
    def gaussian_noise(self, sigma=1.0,size=None, range_min=-0.3, range_max=1.0):
        noise = np.random.random(self.size)
        gaussian_noise = gaussian_filter(noise,sigma) + 0.5*gaussian_filter(noise,sigma/2) + 0.25*gaussian_filter(noise,sigma/4)
        gaussian_noise_min = gaussian_noise.min()
        gaussian_noise_max = gaussian_noise.max()

        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 

        return tex_noise
    
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1,1],image_mask=1):
        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,3))*scale_centroids)


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = np.random.randint(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.1,0.8,size = (num_ellipses,2))],1) #0.1 to 0.8
        random_semi_axes = random_major_axes*random_minor_axes


        # Permuting the axes so that one axes doesn't end up being the major every time.
        rng = np.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)

        
        # Random rotation angles for the ellipsoids
        random_rot_angles = np.random.uniform(size = (num_ellipses,3))*np.pi
        #random_rot_angles = np.zeros(shape = (num_ellipses,3))*np.pi
        
        out = []
        for i in range(num_ellipses):
            out.append(self.ellipsoid(random_centroids[i],*random_semi_axes[i],*random_rot_angles[i],img_dim=self.size) )

        out = np.logical_or.reduce(out)*image_mask
            

        regions = skimage.measure.regionprops(np.int16(out*2))

        if(regions==[]):
            return np.zeros_like(out),-1


        return out,1

    def simulation(self,image,inter_image,image_mask,num_lesions=3):
        param_dict = {}
        
        roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(15))*(image>0.1)
        roi_with_masks = roi
        output_image = image 
        output_mask = np.zeros_like(image_mask)
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        

        num_lesions = np.random.randint(low=num_lesions[0],high=num_lesions[1])
        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0

            x_corr,y_corr,z_corr = np.nonzero(roi_with_masks[:,:,:]*image_mask)
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
            # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
            # We need a loop and random choices tailored here for multiple lesions 

            scale_centroid = np.random.randint(2,self.centroid_scaling)
            num_ellipses = np.random.randint(5,self.num_ellipses)
            semi_axes_range = self.ranges[int(np.random.choice(len(self.ranges),p=self.range_sampling))]

                
            if(semi_axes_range==(2,5)):
                alpha = np.random.uniform(0.5,0.8)
                beta = 1-alpha
            if(semi_axes_range!=(2,5)):
                alpha = np.random.uniform(0.6,0.8)
                beta = 1-alpha


            smoothing_mask = np.random.uniform(0.6,0.8)
            smoothing_image = np.random.uniform(0.3,0.5)

            tex_sigma = np.random.uniform(0.4,0.7)
            range_min = np.random.uniform(-0.5,0.5)
            range_max = np.random.uniform(0.7,1)
            perturb_sigma = np.random.uniform(0.5,5)

            #print(alpha,beta)
            if(semi_axes_range == (2,5)):
                small_sigma = np.random.uniform(1,2)
                out = self.gaussian_small_shapes(image_mask,small_sigma)
            else:   
                out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
                    out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
                

            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = np.random.uniform(1,1.5)
                beta = 1-alpha

                gamma = np.random.uniform(0.5,0.7)
                #print(alpha,beta,gamma)
                out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
                    out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)

            output_mask = np.logical_or(output_mask,out)

            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
            else:
                tex_noise = 1.0
            
            image1 = alpha*output_image + beta*out*tex_noise
            
            # else:
            #     image1 = alpha*output_image*(1-out) + beta*out*tex_noise
            #     image1[image1<0] = 0
            

            output_image = image1
            output_image -= output_image.min()
            output_image /= output_image.max()


            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]
        
        param_dict['num_lesions'] = num_lesions
        output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))
        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    def read_image(self,index,inter_image=False):
        nii_img = nib.load(self.paths[index])
        nii_img_affine = nii_img._affine
        nii_img = nii_img.get_fdata()
        image,img_crop_para = self.tight_crop_data(nii_img)
        image = skiform.resize(image, self.size, order=1, preserve_range=True)
        image -= image.min()
        image /= image.max() + 1e-7
        if(inter_image):
            angle = np.random.uniform(0, 180)
            axes = np.random.choice([0,1,2],2,replace=False)
            input_rotated1 = ndimage.rotate(image, float(angle), axes=axes, reshape=False, mode='nearest')
            image = input_rotated1
        return image,nii_img_affine,img_crop_para
    
    def __getitem__(self, index):
        image,nii_img_affine,img_crop_para = self.read_image(index)
        clean_image = image

        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,nii_img_affine,_ = self.read_image(interpolation_choice,inter_image=True)
        clean_inter_image = inter_image


        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)
            

        else:
            brain_mask_img = ndimage.binary_fill_holes(image>0,structure=np.ones((3,3,3)))
        
        param_dict = {}

        if(self.return_param):
            image, label, param_dict = self.simulation(image,inter_image, brain_mask_img,self.num_lesions)
        else:
            image, label = self.simulation(image,inter_image, brain_mask_img,self.num_lesions)

        return image.astype(np.single),label.astype(np.single),nii_img_affine,

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
