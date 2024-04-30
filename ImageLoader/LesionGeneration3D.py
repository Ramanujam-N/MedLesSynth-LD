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
class LesionGeneration3D(Dataset):
    def __init__(self, img_path, gt_path=None, mask_path = None, roi_path=None, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15):
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

        if(not self.perturb):
            return out,1

        # Perturb in a local neighbhour hood of the lesion
        bounding_box = regions[0].bbox

        noise_b_box = np.random.normal(size = self.size)
        noise_b_box = gaussian_filter(noise_b_box,sigma=perturb_sigma) 
        noise_b_box -= noise_b_box.min()
        noise_b_box /= noise_b_box.max()

        thresholded_b_box = np.zeros_like(out)
        thresholded_b_box[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]] = (noise_b_box>0.6)[bounding_box[0]:bounding_box[3],bounding_box[1]:bounding_box[4],bounding_box[2]:bounding_box[5]]

        labelled_threshold_b_box, nlabels = skimage.measure.label(thresholded_b_box, return_num=True)
        labels_in_big_lesion = np.union1d(out * labelled_threshold_b_box, [])
        labels_tob_removed = np.setdiff1d(np.arange(1, nlabels+1), labels_in_big_lesion)
        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out + labelled_threshold_b_box >0
        
        
        return final_region,1

    def simulation(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None,roi_mask=None):
        param_dict = {}
        
        # roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1) #15
        roi = ndimage.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1) #15
        if(np.array([roi_mask]).any()!=None):
            roi = roi_mask*roi
        roi_with_masks = roi*(1-gt_mask)
        output_image = image
        output_mask = gt_mask
        
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
                alpha = np.random.uniform(0.5,0.6)
                beta = 1-alpha

            if(self.which_data=='lits'):
                alpha = np.random.uniform(0.6,0.8)
                beta = 1-alpha

            smoothing_mask = np.random.uniform(0.6,0.8)
            smoothing_image = np.random.uniform(0.05,0.1) # (0.3,0.5)

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
            if(np.array([roi_mask]).any()!=None):
                out *= roi_mask                

            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
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
            
            if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)
                if(not self.have_noise):
                    tex_noise_edema = 1.0

                smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                
                if(self.which_data == 'lits'):
                    smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                    smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)


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

                output_mask = np.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= output_image.max()
            if(self.have_smoothing and (semi_axes_range==(2,5) or semi_axes_range==(3,5)) or self.which_data=='lits'):
                smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
                smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)

                if(self.which_data=='lits'):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les

                elif(np.random.choice([0,1])*self.dark):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les
                else:
                    image1 = alpha*output_image + beta*smoothed_les
                    image2 = alpha*smoothed_out + beta*smoothed_les
                image1[out>0]=image2[out>0]
                image1[image1<0] =0.1
                
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
        #output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))

        # image_hf = image - gaussian_filter(image,sigma=5)*(image_mask)
        # output_image+=image_hf*(image_mask)
        # output_image -= output_image.min()
        # output_image /= output_image.max()
        # output_image[image_mask==0] = 0

        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    # def simulation_onmask(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None,roi_mask=None):
    #     param_dict = {}
        
    #     # roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1) #15
    #     # roi = ndimage.binary_erosion(image_mask,skimage.morphology.ball(10))*(image>0.1) #15
    #     # roi_with_masks = roi*(1-gt_mask)
    #     output_image = image*(1-gt_mask) + gt_mask*np.mean(image*(gt_mask)*(image>0.15))
    #     output_mask = gt_mask
        
    #     total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
    #                         'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        
    #     labels = skimage.measure.label(gt_mask)
    #     regions = skimage.measure.regionprops(labels)

    #     num_lesions = len(regions)
        
    #     for i in range(num_lesions):
    #         gamma = 0
    #         tex_sigma_edema = 0
    #         # centroid_main = np.array(regions[i].centroid)
    #         centroid_main = np.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T            # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
    #         # We need a loop and random choices tailored here for multiple lesions 

    #         scale_centroid = np.random.randint(2,self.centroid_scaling)
    #         num_ellipses = np.random.randint(5,self.num_ellipses)
    #         semi_axes_range = self.ranges[int(np.random.choice(len(self.ranges),p=self.range_sampling))]

                
    #         if(semi_axes_range==(2,5)):
    #             alpha = np.random.uniform(0.5,0.8)
    #             beta = 1-alpha
    #         if(semi_axes_range!=(2,5)):
    #             alpha = np.random.uniform(0.6,0.8)
    #             beta = 1-alpha

    #         if(self.which_data=='lits'):
    #             alpha = np.random.uniform(0.6,0.8)
    #             beta = 1-alpha

    #         smoothing_mask = np.random.uniform(0.6,0.8)
    #         smoothing_image = np.random.uniform(0.05,0.1) # (0.3,0.5)

    #         tex_sigma = np.random.uniform(0.4,0.7)
    #         range_min = np.random.uniform(-0.5,0.5)
    #         range_max = np.random.uniform(0.7,1)
    #         perturb_sigma = np.random.uniform(0.5,5)

    #         #print(alpha,beta)
    #         if(semi_axes_range == (2,5)):
    #             small_sigma = np.random.uniform(1,2)
    #             out = self.gaussian_small_shapes(image_mask,small_sigma)
    #         else:   
    #             out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
    #             while(shape_status==-1):
    #                 print(shape_status)
    #                 centroid_main = np.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T            # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
    #                 out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
                

    #         if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
    #             semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
    #             tex_sigma_edema = np.random.uniform(1,1.5)
    #             beta = 1-alpha

    #             gamma = np.random.uniform(0.5,0.7)
    #             #print(alpha,beta,gamma)
    #             out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
    #             while(shape_status==-1):
    #                 print(shape_status)
    #                 centroid_main = np.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T            # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
    #                 out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)

    #         output_mask = np.logical_or(output_mask,out)

    #         if(self.have_noise and semi_axes_range !=(2,5)):
    #             tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
    #         else:
    #             tex_noise = 1.0
            
    #         if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5) and self.which_data!='lits'):
    #             tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)
    #             if(not self.have_noise):
    #                 tex_noise_edema = 1.0

    #             smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
    #             smoothed_out = 0.5*output_image + 0.5*inter_image
                
    #             if(self.which_data == 'lits'):
    #                 smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
    #                 smoothed_out = 0.5*output_image + 0.5*inter_image

    #             smoothed_out-=smoothed_out.min()
    #             smoothed_out/=smoothed_out.max()

    #             if(self.which_data=='lits'):
    #                 image1 = alpha*output_image - beta*smoothed_les
    #                 image2 = alpha*smoothed_out - beta*smoothed_les

    #             else:
    #                 image1 = alpha*output_image*(1-(labels==i+1)) + smoothed_les
    #                 image2 = alpha*smoothed_out*(1-(labels==i+1)) + smoothed_les

    #             image1[out_edema>0]=image2[out_edema>0]
    #             image1[image1<0] = 0

    #             output_mask = np.logical_or(output_mask,out_edema)
    #             output_image = image1
    #             output_image -= output_image.min()
    #             output_image /= output_image.max()
    #         if(self.have_smoothing and (semi_axes_range==(2,5) or semi_axes_range==(3,5)) or self.which_data=='lits'):
    #             smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
    #             smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)

    #             if(self.which_data=='lits'):
    #                 image1 = alpha*output_image - beta*smoothed_les
    #                 image2 = alpha*smoothed_out - beta*smoothed_les

    #             elif(np.random.choice([0,1])*self.dark):
    #                 image1 = alpha*output_image - beta*smoothed_les
    #                 image2 = alpha*smoothed_out - beta*smoothed_les
    #             else:
    #                 image1 = alpha*output_image + beta*smoothed_les
    #                 image2 = alpha*smoothed_out + beta*smoothed_les
    #             image1[out>0]=image2[out>0]
    #             image1[image1<0] =0.1
                
    #         # else:
    #         #     image1 = alpha*output_image*(1-out) + beta*out*tex_noise
    #         #     image1[image1<0] = 0
            

    #         output_image = image1
    #         output_image -= output_image.min()
    #         output_image /= output_image.max()


    #         # roi_with_masks *= (1-output_mask)>0
            
    #         total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
    #                         tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
    #         for j in range(len(total_params)):
    #             param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]
        
    #     param_dict['num_lesions'] = num_lesions
    #     #output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))

    #     if(self.return_param):
    #         return output_image, output_mask, param_dict
    #     else:
    #         return output_image, output_mask

    def simulation_onmask(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None,roi_mask=None):
        param_dict = {}
        output_total_mask=gt_mask
        output_total_mask = output_total_mask>0
        mask_label,num_lesions = skimage.measure.label(output_total_mask>0,return_num=True)
        regions = list(skimage.measure.regionprops(mask_label))

        roi = skimage.morphology.binary_erosion(image_mask,skimage.morphology.ball(15))*(image>0.1)
        roi_with_masks = roi
        output_image = image 
        output_mask = np.zeros_like(image_mask)
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']

        for i in range(0,num_lesions):
            # print(num_lesions)
            gamma = 0
            tex_sigma_edema = 0

            centroid_main = np.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T
            
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

            out*=output_total_mask
                #pass
            # sumout = np.sum(np.sum(out, axis=0), axis=0)
            # slide_no = np.where(sumout == np.amax(sumout))[0][0]
            # plt.imshow(out[:,:,slide_no])
            # plt.show()
            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = np.random.uniform(1,1.5)
                beta = 1-alpha

                gamma = np.random.uniform(0.5,0.7)
                #print(alpha,beta,gamma)
                out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                out_edema*=output_total_mask
                    #pass

            output_mask = np.logical_or(output_mask,out)

            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
            else:
                tex_noise = 1.0
            
            if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)

                smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                
                smoothed_out-=smoothed_out.min()
                smoothed_out/=smoothed_out.max()


                image1 = alpha*output_image + smoothed_les
                image2 = alpha*smoothed_out + smoothed_les

                image1[out_edema>0]=image2[out_edema>0]
                image1[image1<0] = 0

                output_mask = np.logical_or(output_mask,out_edema)
                output_image = image1
                output_image -= output_image.min()
                output_image /= output_image.max()
            if(self.have_smoothing and semi_axes_range==(2,5) or semi_axes_range==(3,5)):
                smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
                smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                

                if(np.random.choice([0,1])*self.dark):
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
        output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))
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

        if(inter_image and self.which_data!='lits'):
            angle = np.random.uniform(0, 180)
            axes = np.random.choice([0,1,2],2,replace=False)
            input_rotated1 = ndimage.rotate(image, float(angle), axes=axes, reshape=False, mode='nearest')
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
            brain_mask_img = skiform.resize(brain_mask_img, self.size, order=0, preserve_range=True)
            
        else:
            brain_mask_img = ndimage.binary_fill_holes(image>0,structure=np.ones((3,3,3)))
        
        param_dict = {}

        if(self.return_param):
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




#########################################################
# New Setting 
    # def simulation_with_maskingregion_centroid(self,image,inter_image,brain_mask_img,output_total_mask,num_lesions=3):
    #     param_dict = {}

    #     if(self.which_data=='wmh'):
    #         num_lesions = np.random.randint(1,30) #5,15
    #         ranges = [(2,5),(3,5)]
    #         centroid_scaling = 20
    #     elif(self.which_data=='brats'):
    #         num_lesions = np.random.randint(1,5)
    #         #ranges = [(5,10),(10,15)]
    #         ranges = [(15,20),(15,20)]
    #         centroid_scaling = 15
    #     elif(self.which_data=='size1'):
    #         num_lesions = np.random.randint(1,30) #5,15
    #         ranges = [(2,5),(2,5)]
    #         centroid_scaling = 20

    #     elif(self.which_data=='size2'):
    #         num_lesions = np.random.randint(1,15) #5,15
    #         ranges = [(3,5),(3,5)]
    #         centroid_scaling = 20

    #     elif(self.which_data=='size3'):
    #         num_lesions = np.random.randint(1,10) #5,15
    #         ranges = [(5,10),(5,10)]
    #         centroid_scaling = 15

    #     elif(self.which_data=='size4'):
    #         num_lesions = np.random.randint(1,5) #5,15
    #         ranges = [(10,15),(10,15)]
    #         centroid_scaling = 15

    #     elif(self.which_data=='all'):
    #         num_lesions = np.random.randint(1,30) #5,15
    #         ranges = [(2,5),(3,5),(5,10),(10,15)]
    #         centroid_scaling = 20
    #     else:
    #         num_lesions = np.random.randint(1,5)
    #         ranges = [(5,10),(10,15)]
    #         centroid_scaling = 15
        
        
    #     output_total_mask*=brain_mask_img
    #     output_total_mask = output_total_mask>0
    #     mask_label,num_lesions = skimage.measure.label(output_total_mask>0,return_num=True)
    #     regions = list(skimage.measure.regionprops(mask_label))

    #     roi = skimage.morphology.binary_erosion(brain_mask_img,skimage.morphology.ball(15))*(image>0.1)
    #     roi_with_masks = roi
    #     output_image = image 
    #     output_mask = np.zeros_like(brain_mask_img)
        
    #     total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
    #                         'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']

    #     for i in range(0,num_lesions):
    #         # print(num_lesions)
    #         gamma = 0
    #         tex_sigma_edema = 0

    #         centroid_main = np.array([regions[i].centroid[1],regions[i].centroid[0],regions[i].centroid[2]]).T
            
    #         # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
    #         # We need a loop and random choices tailored here for multiple lesions 

    #         scale_centroid = np.random.randint(2,centroid_scaling)
    #         num_ellipses = 15
    #         if(self.which_data=='all'):
    #             semi_axes_range = ranges[int(np.random.choice(4,p=[0.35,0.4,0.15,0.1]))]
    #         else:
    #             semi_axes_range = ranges[int(np.random.choice(2,p=[0.7,0.3]))]
                
            
    #         if(semi_axes_range==(2,5)):
    #             alpha = np.random.uniform(0.5,0.8)
    #             beta = 1-alpha
    #         if(semi_axes_range!=(2,5)):
    #             alpha = np.random.uniform(0.6,0.8)
    #             beta = 1-alpha


    #         smoothing_mask = np.random.uniform(0.6,0.8)
    #         smoothing_image = np.random.uniform(0.3,0.5)

    #         tex_sigma = np.random.uniform(0.4,0.7)
    #         range_min = np.random.uniform(-0.5,0.5)
    #         range_max = np.random.uniform(0.7,1)
    #         perturb_sigma = np.random.uniform(0.5,5)

    #         #print(alpha,beta)
    #         if(semi_axes_range == (2,5)):
    #             small_sigma = np.random.uniform(1,2)
    #             out = self.gaussian_small_shapes(brain_mask_img,small_sigma)
    #         else:   
    #             out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=brain_mask_img) 

    #         if(self.on_real_with_mask):
    #             out*=output_total_mask
    #             #pass
    #         # sumout = np.sum(np.sum(out, axis=0), axis=0)
    #         # slide_no = np.where(sumout == np.amax(sumout))[0][0]
    #         # plt.imshow(out[:,:,slide_no])
    #         # plt.show()
    #         if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
    #             semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
    #             tex_sigma_edema = np.random.uniform(1,1.5)
    #             beta = 1-alpha

    #             gamma = np.random.uniform(0.5,0.7)
    #             #print(alpha,beta,gamma)
    #             out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=brain_mask_img)
    #             if(self.on_real_with_mask):
    #                 out_edema*=output_total_mask
    #                 #pass

    #         output_mask = np.logical_or(output_mask,out)

    #         if(self.have_noise and semi_axes_range !=(2,5)):
    #             tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
    #         else:
    #             tex_noise = 1.0
            
    #         if(self.have_smoothing and self.have_edema and semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
    #             tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)

    #             smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
    #             smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                
    #             smoothed_out-=smoothed_out.min()
    #             smoothed_out/=smoothed_out.max()


    #             image1 = alpha*output_image + smoothed_les
    #             image2 = alpha*smoothed_out + smoothed_les

    #             image1[out_edema>0]=image2[out_edema>0]
    #             image1[image1<0] = 0

    #             output_mask = np.logical_or(output_mask,out_edema)
    #             output_image = image1
    #             output_image -= output_image.min()
    #             output_image /= output_image.max()
    #         if(self.have_smoothing and semi_axes_range==(2,5) or semi_axes_range==(3,5)):
    #             smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
    #             smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                

    #             if(np.random.choice([0,1])*self.dark):
    #                 image1 = alpha*output_image - beta*smoothed_les
    #                 image2 = alpha*smoothed_out - beta*smoothed_les
    #             else:
    #                 image1 = alpha*output_image + beta*smoothed_les
    #                 image2 = alpha*smoothed_out + beta*smoothed_les
    #             image1[out>0]=image2[out>0]
    #             image1[image1<0] = 0.1
    #         # else:
    #         #     image1 = alpha*output_image*(1-out) + beta*out*tex_noise
    #         #     image1[image1<0] = 0
            

    #         output_image = image1
    #         output_image -= output_image.min()
    #         output_image /= output_image.max()

    #         # if(int(np.random.choice([0,1]))):
    #         #     output_image[out>0] = 1-output_image[out>0]
    #         roi_with_masks *= (1-output_mask)>0
            
    #         total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
    #                         tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
    #         for j in range(len(total_params)):
    #             param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]

    #     param_dict['num_lesions'] = num_lesions
    #     output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))
    #     if(self.return_param):
    #         return output_image, output_mask, param_dict
    #     else:
    #         return output_image, output_mask