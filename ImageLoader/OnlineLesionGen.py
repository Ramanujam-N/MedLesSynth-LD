# import skimage
# import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
# import skimage.morphology
# import skimage.transform as skiform
# from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# from scipy import ndimage
# import scipy as sc
# from scipy.stats import rice
# from scipy.ndimage import gaussian_filter
import cupy as cp
import cupyx as cpx
import cucim
class OnlineLesionGen(Dataset):
    def __init__(self, img_path, gt_path=None, mask_path = None, roi_path=None, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15,rician=False):
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
        self.use_rician = rician
    def normalize(self,image,range_min=0,range_max=1):
        image_min = image.min()
        image_max = image.max()
        range_max = cp.random.uniform(range_max-0.5*range_max,range_max)
        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        out = ((image - image_min)*(range_max-range_min)/(image_max-image_min)) + range_min 
        return out

    def gaussian_small_shapes(self,image_mask,small_sigma = [1,1,1]):

        image_mask = cpx.scipy.ndimage.binary_erosion(image_mask,cucim.skimage.morphology.cube(5))
        index = -1
        while(index==-1):
            noise = cp.random.normal(size = image_mask.shape)
            smoothed_noise = cucim.skimage.filters.gaussian(noise,sigma=small_sigma) #+ cucim.skimage.filters.gaussian(noise,sigma=[4,4,4]) + cucim.skimage.filters.gaussian(noise,sigma=[3,3,3]) + cucim.skimage.filters.gaussian(noise,sigma=[2,2,2]) + 0.1*cucim.skimage.filters.gaussian(noise,sigma=[1,1,1])
            smoothed_noise -= smoothed_noise.min()
            smoothed_noise /= smoothed_noise.max()

            bg_mask = (smoothed_noise>0.3)*(image_mask)
            mask = (1-bg_mask)*(image_mask)

            labelled = cpx.scipy.ndimage.label(mask*2,background = 0)         
            regions = cucim.skimage.measure.regionprops(labelled)

            total_brain_area = cp.sum(image_mask)
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
        
    def ellipsoid(self,coord=(1,2,1),semi_a = 4, semi_b = 34, semi_c = 34, alpha=cp.pi/4, beta=cp.pi/4, gamma=cp.pi/4,img_dim=(128,128,128)):
        img_dim = list(img_dim)
        img_dim[0],img_dim[1] = img_dim[1],img_dim[0]
        x = cp.linspace(-img_dim[0]//2,cp.ceil(img_dim[0]//2),img_dim[0])
        y = cp.linspace(-img_dim[1]//2,cp.ceil(img_dim[1]//2),img_dim[1])
        z = cp.linspace(-img_dim[2]//2,cp.ceil(img_dim[2]//2),img_dim[2])
        x,y,z = cp.meshgrid(x,y,z)

        # Take the centering into effect   
        x=(x - coord[0] + img_dim[0]//2)
        y=(y - coord[1] + img_dim[1]//2)
        z=(z - coord[2] + img_dim[2]//2)

        ellipsoid_std_axes = cp.stack([x,y,z],0)

        alpha = -alpha
        beta = -beta
        gamma = -gamma    

        rotation_x = cp.array([[1, 0, 0],
                                [0, cp.cos(alpha), -cp.sin(alpha)],
                                [0, cp.sin(alpha), cp.cos(alpha)]])
        
        rotation_y = cp.array([[cp.cos(beta), 0, cp.sin(beta)],
                                [0, 1, 0],
                                [-cp.sin(beta), 0, cp.cos(beta)]])
        
        rotation_z = cp.array([[cp.cos(gamma), -cp.sin(gamma), 0],
                                [cp.sin(gamma), cp.cos(gamma), 0],
                                [0, 0, 1]])

        rot_matrix = rotation_x@rotation_y@rotation_z
        ellipsoid_rot_axes = cp.tensordot(rot_matrix,ellipsoid_std_axes,axes=([1,0]))

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
        noise = cp.random.random(size)
        gaussian_noise = cucim.skimage.filters.gaussian(noise,sigma) + 0.5*cucim.skimage.filters.gaussian(noise,sigma/2) + 0.25*cucim.skimage.filters.gaussian(noise,sigma/4)
        gaussian_noise_min = gaussian_noise.min()
        gaussian_noise_max = gaussian_noise.max()

        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 

        return tex_noise
        
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1,1],image_mask=1):
        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main.T + (cp.random.random((num_ellipses,3))*scale_centroids)


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = cp.random.uniform(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = cp.concatenate([cp.ones((num_ellipses,1)),cp.random.uniform(0.3,1,size = (num_ellipses,2))],1) #0.1 to 0.8
        random_semi_axes = random_major_axes*random_minor_axes


        # Permuting the axes so that one axes doesn't end up being the major every time.
        rng = cp.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)

        
        # Random rotation angles for the ellipsoids
        random_rot_angles = cp.random.uniform(-cp.pi/2,cp.pi/2,size = (num_ellipses,3))
        #random_rot_angles = cp.zeros(shape = (num_ellipses,3))*cp.pi
        
        out = []
        for i in range(num_ellipses):
            out.append(self.ellipsoid(random_centroids[i],*random_semi_axes[i],*random_rot_angles[i],img_dim=image_mask.shape) )

        out = cp.logical_or.reduce(out)*image_mask
            

        regions = cucim.skimage.measure.regionprops(cp.int16(out*2))

        if(regions==[]):
            return cp.zeros_like(out),-1

        if(not self.perturb):
            return out,1

        # Perturb in a local neighbhour hood of the lesion
        bounding_box = regions[0].bbox

        noise_b_box = cp.random.normal(size = image_mask.shape)
        noise_b_box = cucim.skimage.filters.gaussian(noise_b_box,sigma=perturb_sigma) 
        noise_b_box -= noise_b_box.min()
        noise_b_box /= noise_b_box.max()

        thresholded_b_box = cp.zeros_like(out)
        thresholded_b_box[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]] = (noise_b_box>0.6)[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]]

        labelled_threshold_b_box, nlabels = cpx.scipy.ndimage.label(thresholded_b_box, return_num=True)
        labels_in_big_lesion = cp.union1d(out * labelled_threshold_b_box, [])
        labels_tob_removed = cp.setdiff1d(cp.arange(1, nlabels+1), labels_in_big_lesion)
        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out + labelled_threshold_b_box >0
        
        
        return final_region,1

    def simulation(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None,roi_mask=None):
        param_dict = {}
        if(image.shape !=inter_image.shape):
            inter_image = cucim.skimage.transform.resize(image, image.shape, order=1, preserve_range=True ,anti_aliasing=True)
        # roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1) #15
        roi = cucim.skimage.morphology.binary_erosion(image_mask,cucim.skimage.morphology.ball(5))*(image>0.1) # 12 #15

        if(cp.array([roi_mask]).any()!=None):
            roi = roi_mask*roi
        roi_with_masks = roi*(1-gt_mask)
        output_image = image
        output_mask = gt_mask
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        

        num_lesions = cp.random.randint(low=num_lesions[0],high=num_lesions[1])
        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0
            x_corr,y_corr,z_corr = cp.nonzero(roi_with_masks[:,:,:]*image_mask)
            random_coord_index = cp.random.choice(len(x_corr),1)
            centroid_main = cp.array([y_corr[random_coord_index],x_corr[random_coord_index],z_corr[random_coord_index]])

            # We need a loop and random choices tailored here for multiple lesions 

            scale_centroid = cp.random.uniform(2,self.centroid_scaling)
            num_ellipses = cp.random.randint(5,self.num_ellipses)
            semi_axes_range = self.ranges[int(cp.random.choice(len(self.ranges),p=self.range_sampling))]

                
            if(semi_axes_range==(2,5)):
                alpha = cp.random.uniform(0.7,0.8)
                beta = 1-alpha
            if(semi_axes_range!=(2,5)):
                alpha = cp.random.uniform(0.7,0.9)
                beta = 1-alpha

            if(self.which_data=='lits'):
                alpha = cp.random.uniform(0.75,0.9)
                beta = 1-alpha

            if(self.which_data=='wmh'):
                alpha = cp.random.uniform(0.7,0.8)
                beta = 1-alpha

            smoothing_mask = cp.random.uniform(0.6,0.8)
            smoothing_image = cp.random.uniform(0.05,0.1) # (0.3,0.5)

            if(self.which_data=='lits'):
                smoothing_mask = cp.random.uniform(0.2,0.3)
                
            tex_sigma = cp.random.uniform(0.4,0.7)
            range_min = cp.random.uniform(0,0.5)
            range_max = cp.random.uniform(0.7,1)
            perturb_sigma = cp.random.uniform(0.5,5)

            #print(alpha,beta)
            if(semi_axes_range == (2,5)):
                small_sigma = cp.random.uniform(1,2)
                out = self.gaussian_small_shapes(image_mask,small_sigma)
            else:   
                out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = cp.random.choice(len(x_corr),1)
                    centroid_main = cp.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
                    out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
            if(cp.array([roi_mask]).any()!=None):
                out *= roi_mask                

            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = cp.random.uniform(1,1.5)
                beta = 1-alpha

                gamma = cp.random.uniform(0.5,0.7)
                #print(alpha,beta,gamma)
                out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = cp.random.choice(len(x_corr),1)
                    centroid_main = cp.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
                    out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)

            output_mask = cp.logical_or(output_mask,out)

            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise = self.gaussian_noise(tex_sigma,image_mask.shape,range_min,range_max)
            else:
                tex_noise = 1.0
            
            if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema,image_mask.shape,range_min,range_max)
                if(not self.have_noise):
                    tex_noise_edema = 1.0

                smoothed_les = beta*cucim.skimage.filters.gaussian(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = cucim.skimage.filters.gaussian(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                
                if(self.which_data == 'lits'):
                    smoothed_les = beta*cucim.skimage.filters.gaussian(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                    smoothed_out = cucim.skimage.filters.gaussian(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)


                smoothed_out-=smoothed_out.min()
                smoothed_out/=smoothed_out.max()

                if(self.which_data=='lits'):
                    
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les

                else:
                    image1 = alpha*output_image + smoothed_les
                    image2 = alpha*smoothed_out + smoothed_les

                image1[out_edema>0]=image2[out_edema>0]
                image1[image1<0] = 0

                output_mask = cp.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= output_image.max()
            if(self.have_smoothing and (semi_axes_range==(2,5) or semi_axes_range==(3,5)) or self.which_data=='lits'):
                smoothed_les = cucim.skimage.filters.gaussian(out*tex_noise, sigma=smoothing_mask)
                if(self.which_data!='wmh'):
                    smoothed_les = out*(tex_noise + output_image)
                smoothed_out = cucim.skimage.filters.gaussian(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)

                if(self.which_data=='lits'):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les
                
                elif(cp.random.choice([0,1])*self.dark):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les
                else:
                    image1 = alpha*output_image + beta*smoothed_les
                    image2 = alpha*smoothed_out + beta*smoothed_les
                image1[out>0]=image2[out>0]
                image1[image1<0] =cp.random.uniform(0,0.1)
                print(image1.min())

            # else:
            #     image1 = alpha*output_image*(1-out) + beta*out*tex_noise
            #     image1[image1<0] = 0
            image_stuff = image>0.01
            image1[image_stuff==0] = 0
            output_image = image1
            output_image -= output_image.min()
            output_image /= output_image.max()


            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]
        
        param_dict['num_lesions'] = num_lesions
        
        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    def simulation_onmask(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None,roi_mask=None):
        param_dict = {}
        output_total_mask=gt_mask
        output_total_mask = output_total_mask>0
        mask_label,num_lesions = cucim.skimage.measure.label(output_total_mask>0,return_num=True)
        regions = list(cucim.skimage.measure.regionprops(mask_label))

        roi = cucim.skimage.morphology.binary_erosion(image_mask,cucim.skimage.morphology.ball(15))*(image>0.1)
        roi_with_masks = roi
        output_image = image 
        output_mask = cp.zeros_like(image_mask)
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']

        for i in range(0,num_lesions):
            # print(num_lesions)
            gamma = 0
            tex_sigma_edema = 0

            centroid_main = cp.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T
            
            # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
            # We need a loop and random choices tailored here for multiple lesions 

            scale_centroid = cp.random.randint(2,self.centroid_scaling)
            num_ellipses = cp.random.randint(5,self.num_ellipses)
            semi_axes_range = self.ranges[int(cp.random.choice(len(self.ranges),p=self.range_sampling))]
            
            if(semi_axes_range==(2,5)):
                alpha = cp.random.uniform(0.5,0.8)
                beta = 1-alpha
            if(semi_axes_range!=(2,5)):
                alpha = cp.random.uniform(0.6,0.8)
                beta = 1-alpha


            smoothing_mask = cp.random.uniform(0.6,0.8)
            smoothing_image = cp.random.uniform(0.3,0.5)

            tex_sigma = cp.random.uniform(0.4,0.7)
            range_min = cp.random.uniform(-0.5,0.5)
            range_max = cp.random.uniform(0.7,1)
            perturb_sigma = cp.random.uniform(0.5,5)

            #print(alpha,beta)
            if(semi_axes_range == (2,5)):
                small_sigma = cp.random.uniform(1,2)
                out = self.gaussian_small_shapes(image_mask,small_sigma)
            else:   
                out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)

            out*=output_total_mask
                #pass
            # sumout = cp.sum(np.sum(out, axis=0), axis=0)
            # slide_no = cp.where(sumout == cp.amax(sumout))[0][0]
            # plt.imshow(out[:,:,slide_no])
            # plt.show()
            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = cp.random.uniform(1,1.5)
                beta = 1-alpha

                gamma = cp.random.uniform(0.5,0.7)
                #print(alpha,beta,gamma)
                out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                out_edema*=output_total_mask
                    #pass

            output_mask = cp.logical_or(output_mask,out)

            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise = self.gaussian_noise(tex_sigma,image_mask.shape,range_min,range_max)
            else:
                tex_noise = 1.0
            
            if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema,image_mask.shape,range_min,range_max)

                smoothed_les = beta*cucim.skimage.filters.gaussian(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = cucim.skimage.filters.gaussian(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                
                smoothed_out-=smoothed_out.min()
                smoothed_out/=smoothed_out.max()


                image1 = alpha*output_image + smoothed_les
                image2 = alpha*smoothed_out + smoothed_les

                image1[out_edema>0]=image2[out_edema>0]
                image1[image1<0] = 0

                output_mask = cp.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= output_image.max()
            if(self.have_smoothing and semi_axes_range==(2,5) or semi_axes_range==(3,5)):
                smoothed_les = cucim.skimage.filters.gaussian(out*tex_noise, sigma=smoothing_mask)
                smoothed_out = cucim.skimage.filters.gaussian(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                

                if(cp.random.choice([0,1])*self.dark):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les
                else:
                    image1 = alpha*output_image + beta*smoothed_les
                    image2 = alpha*smoothed_out + beta*smoothed_les
                image1[out>0]=image2[out>0]
                image1[image1<0] = 0.1
            # else:
            #     image1 = alpha*output_image*(1-out) + beta*out*tex_noise
            #     image1[image1<0] = 0
            

            output_image = image1
            output_image -= output_image.min()
            output_image /= output_image.max()

            # if(int(np.random.choice([0,1]))):
            #     output_image[out>0] = 1-output_image[out>0]
            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]

        param_dict['num_lesions'] = num_lesions
        output_mask = cucim.skimage.morphology.binary_dilation(output_mask,cucim.skimage.morphology.cube(2))
        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask


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
            roi_mask = cucim.skimage.morphology.binary_dilation(roi_mask)

            gt_mask = cp.zeros_like(image)
        else:
            gt_mask = cp.zeros_like(image)
            roi_mask = None

        if(inter_image and self.which_data!='lits'):
            angle = cp.random.uniform(0, 180)
            axes = cp.random.choice([0,1,2],2,replace=False)
            input_rotated1 = cpx.scipy.ndimage.rotate(image, float(angle), axes=axes, reshape=False, mode='nearest')
            image = input_rotated1
        
        return image,nii_img_affine,img_crop_para,gt_mask,roi_mask
    
    def __getitem__(self, index):
        image,nii_img_affine,img_crop_para,gt_mask,roi_mask = self.read_image(index)
        clean_image = image

        interpolation_choice = cp.random.choice(len(self.paths))
        inter_image,nii_img_affine,_,_,_ = self.read_image(interpolation_choice,inter_image=True)
        clean_inter_image = inter_image


        if(self.mask_path):
            brain_mask_img = nib.load(self.mask_path[index]).get_fdata()
            brain_mask_img = brain_mask_img[img_crop_para[0]:img_crop_para[0] + img_crop_para[1],img_crop_para[2]:img_crop_para[2] + img_crop_para[3],img_crop_para[4]:img_crop_para[4] + img_crop_para[5]]
            # brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)
            
        else:
            brain_mask_img = cpx.scipy.ndimage.binary_fill_holes(image>0,structure=cp.ones((3,3,3)))
        
        param_dict = {}

        if(self.return_param):
            image, label, param_dict = self.simulation(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
            # image, label, param_dict = self.simulation_onmask(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
        else:
            image, label = self.simulation(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)
            # image, label = self.simulation_onmask(image,inter_image, brain_mask_img,self.num_lesions,gt_mask=gt_mask,roi_mask=roi_mask)

        return image.astype(cp.single),label.astype(cp.single),nii_img_affine,self.paths[index],param_dict

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

        row_sum = cp.sum(cp.sum(img_data, axis=1), axis=1)
        col_sum = cp.sum(cp.sum(img_data, axis=0), axis=1)
        stack_sum = cp.sum(cp.sum(img_data, axis=1), axis=0)
        rsid, reid, rlen = self.cut_zeros1d(row_sum)
        csid, ceid, clen = self.cut_zeros1d(col_sum)
        ssid, seid, slen = self.cut_zeros1d(stack_sum)
        return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], [rsid, rlen, csid, clen, ssid, slen]
