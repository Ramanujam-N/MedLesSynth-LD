import skimage
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import skimage.morphology
import skimage.transform as skiform
from scipy.ndimage import gaussian_filter
from scipy import ndimage


def normalize(image):
    image-=image.min()
    image/=image.max() + 1e-7
    return image

class LesionGeneration2D(Dataset):
    def __init__(self, img_path, gt_path=None,type_of_imgs='png',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15):
        self.paths = img_path
        self.gt_path = gt_path
        self.transform = transform
        self.size = size
        self.have_texture = have_texture
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
    def gaussian_small_shapes(self,image_mask,small_sigma = [1,1]):

        image_mask = skimage.morphology.binary_erosion(image_mask,skimage.morphology.square(5))
        index = -1
        while(index==-1):
            noise = np.random.normal(size = self.size)
            smoothed_noise = gaussian_filter(noise,sigma=small_sigma) + gaussian_filter(noise,sigma=[4,4,4]) + gaussian_filter(noise,sigma=[3,3,3]) + gaussian_filter(noise,sigma=[2,2,2]) + 0.1*gaussian_filter(noise,sigma=[1,1,1])
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
        
    def ellipsoid(self,coord=(1,2),semi_a = 4, semi_b = 34, alpha=np.pi/4, beta=np.pi/4,img_dim=(128,128)):
        x = np.linspace(-img_dim[0]//2,np.ceil(img_dim[0]//2),img_dim[0])
        y = np.linspace(-img_dim[1]//2,np.ceil(img_dim[1]//2),img_dim[1])
 
        y,x = np.meshgrid(y,x) # x,y to y,x 


        # Take the centering into effect   
        x=(x - coord[0] + img_dim[0]//2)
        y=(y - coord[1] + img_dim[1]//2)

        ellipsoid_std_axes = np.stack([x,y],0) 

        alpha = -alpha
        beta = -beta

        rotation_x = np.array([[np.cos(alpha), -np.sin(alpha)],
                                [np.sin(alpha), np.cos(alpha)]])
        
        rotation_y = np.array([[np.cos(beta), np.sin(beta)],
                                [-np.sin(beta), np.cos(beta)]])
        


        rot_matrix = rotation_x@rotation_y
        ellipsoid_rot_axes = np.tensordot(rot_matrix,ellipsoid_std_axes,axes=([1,0]))

        x,y = ellipsoid_rot_axes
        x**=2
        y**=2

        a = semi_a**2
        b = semi_b**2

        ellipsoid = x/a + y/b - 1
        ellipsoid = ellipsoid<0 

        return ellipsoid
        
    def gaussian_noise(self, sigma=1.0, size = (128,128), range_min=-0.3, range_max=1.0):
        noise = np.random.random(size)
        gaussian_noise = gaussian_filter(noise,sigma) + 0.5*gaussian_filter(noise,sigma/2) + 0.25*gaussian_filter(noise,sigma/4)
        gaussian_noise_min = gaussian_noise.min()
        gaussian_noise_max = gaussian_noise.max()

        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 

        return tex_noise
    
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1],image_mask=1):
        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,2))*scale_centroids)


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = np.random.randint(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.5,0.8,size = (num_ellipses,1))],1)
        random_semi_axes = random_major_axes*random_minor_axes

        # Permuting the axes so that one axes doesn't end up being the major every time.
        rng = np.random.default_rng()
        random_semi_axes = rng.permuted(random_semi_axes, axis=1)

        
        # Random rotation angles for the ellipsoids
        random_rot_angles = np.random.uniform(size = (num_ellipses,2))*np.pi
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
        thresholded_b_box[bounding_box[0]:bounding_box[2],bounding_box[1]:bounding_box[3]] = (noise_b_box>0.6)[bounding_box[0]:bounding_box[2],bounding_box[1]:bounding_box[3]]

        labelled_threshold_b_box, nlabels = skimage.measure.label(thresholded_b_box, return_num=True)
        labels_in_big_lesion = np.union1d(out * labelled_threshold_b_box, [])
        labels_tob_removed = np.setdiff1d(np.arange(1, nlabels+1), labels_in_big_lesion)
        for i in labels_tob_removed:
            labelled_threshold_b_box[labelled_threshold_b_box == i] = 0

        final_region = out + labelled_threshold_b_box >0
        
        
        return final_region,1

    def simulation(self,image,inter_image,image_mask,num_lesions=3,gt_mask=None):
        param_dict = {}
        
        roi = image_mask

        roi_with_masks = roi*(1-gt_mask)
        output_image = image 
        output_mask = gt_mask
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        

        for i in range(num_lesions):
            gamma = 0
            tex_sigma_edema = 0
            print(image_mask.shape)
            x_corr,y_corr = np.nonzero(roi_with_masks[:,:]*image_mask)
            random_coord_index = np.random.choice(len(x_corr),1)
            centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index]])
            # We need a loop and random choices tailored here for multiple lesions 

            scale_centroid = np.random.randint(2,self.centroid_scaling)
            num_ellipses = np.random.randint(1,self.num_ellipses)

            semi_axes_range = self.ranges[int(np.random.choice(len(self.ranges),p=self.range_sampling))]

            if(semi_axes_range==(2,5)):
                alpha = np.random.uniform(0.5,0.8)
                beta = 1-alpha
            if(semi_axes_range!=(2,5)):
                alpha = np.random.uniform(0.6,0.8)
                beta = 1-alpha

            if(self.which_data=='busi'):
                alpha = np.random.uniform(0.8,0.9) # 0.4 0.5
                beta = 1-alpha

            smoothing_mask = np.random.uniform(0.6,0.8)
            if(self.which_data=='busi'):
                smoothing_mask = np.random.uniform(0.6,0.8)

            smoothing_image = np.random.uniform(0.3,0.5)

            tex_sigma = np.random.uniform(0.4,0.7)
            range_min = np.random.uniform(-0.5,0.5)
            range_max = np.random.uniform(0.7,1)
            perturb_sigma = np.random.uniform(0.5,5)

            if(semi_axes_range == (2,5)):
                small_sigma = np.random.uniform(1,2)
                out = self.gaussian_small_shapes(image_mask,small_sigma)
            else:   
                out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index]])
                    out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
                
            if(semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):
                semi_axes_range_edema = (semi_axes_range[0]+5,semi_axes_range[1]+5)
                tex_sigma_edema = np.random.uniform(1,1.5)
                beta = 1-alpha

                gamma = np.random.uniform(0.5,0.7)
                out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)
                while(shape_status==-1):
                    print(shape_status)
                    random_coord_index = np.random.choice(len(x_corr),1)
                    centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index]])
                    out_edema,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range_edema,perturb_sigma,image_mask=image_mask)

            output_mask = np.logical_or(output_mask,out)

            if(self.have_noise and semi_axes_range !=(2,5)):
                tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
            else:
                tex_noise = 1.0
            
            if(self.have_smoothing and semi_axes_range !=(2,5) and semi_axes_range !=(3,5)):

                if(self.have_edema):
                    tex_noise_edema = self.gaussian_noise(tex_sigma_edema,self.size,range_min,range_max)

                    smoothed_les = beta*gaussian_filter(out_edema*(gamma*(1-out)*tex_noise_edema + 4*gamma*out*tex_noise), sigma=smoothing_mask)
                    smoothed_out = gaussian_filter(0.5*output_image + 0.5*inter_image, sigma=smoothing_image)
                else:
                    smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
                    if(self.have_texture):
                        smoothed_out = output_image #+ 0.5*inter_image, sigma=smoothing_image)
                    else:
                        smoothed_out = output_image

                smoothed_out-=smoothed_out.min()
                smoothed_out/=smoothed_out.max()

                if(self.which_data=='busi'):
                    image1 = alpha*output_image*(1-out) + beta*smoothed_les
                    image2 = alpha*smoothed_out*(1-out) + beta*smoothed_les
                    
                    
                else:
                    image1 = alpha*output_image + smoothed_les
                    image2 = alpha*smoothed_out + smoothed_les

                if(self.have_edema):
                    output_mask = np.logical_or(output_mask,out_edema)
                    image1[out_edema>0]=image2[out_edema>0]
                else:
                    image1[out>0]=image2[out>0]

                image1[image1<0] = 0
                # image1[out>0] = ndimage.grey_erosion(image1*(out>0),structure = np.ones((1,5)))[out>0]
                dilated_out = ndimage.binary_dilation(out>0,structure=np.ones((4,4)))
                image1[dilated_out>0] = gaussian_filter(image1[dilated_out>0],sigma=3)
                
            
            if(self.have_smoothing and semi_axes_range==(2,5) or semi_axes_range==(3,5)):
                smoothed_les = gaussian_filter(out*tex_noise, sigma=smoothing_mask)
                if(not self.have_texture):
                    smoothed_out = output_image
                else:
                    smoothed_out = output_image
                
                if(self.which_data=='busi'):
                    image1 = alpha*output_image*(1-out) + beta*smoothed_les
                    image2 = alpha*smoothed_out*(1-out) + beta*smoothed_les

                elif(np.random.choice([0,1])*self.dark):
                    image1 = alpha*output_image - beta*smoothed_les
                    image2 = alpha*smoothed_out - beta*smoothed_les
                else:
                    image1 = alpha*output_image + beta*smoothed_les
                    image2 = alpha*smoothed_out + beta*smoothed_les
                
                image1[out>0]=image2[out>0]
                image1[image1<0] = 0
                # image1[out>0] = ndimage.grey_erosion(image1*(out>0),structure = np.ones((1,5)))[out>0]
                dilated_out = ndimage.binary_dilation(out>0,structure=np.ones((10,10)))
                image1[dilated_out>0] = gaussian_filter(image1[dilated_out>0],sigma=3)

            output_image = image1
            output_image -= output_image.min()
            output_image /= output_image.max()

            roi_with_masks *= (1-output_mask)>0
            
            total_params = [scale_centroid,num_ellipses,semi_axes_range,alpha,beta,gamma,smoothing_mask,
                            tex_sigma,range_min,range_max,tex_sigma_edema,perturb_sigma]
            
            for j in range(len(total_params)):
                param_dict[str(i)+'_'+total_param_list[j]] = total_params[j]
        
        param_dict['num_lesions'] = num_lesions
        #output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.square(2))
        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    def read_image(self,index):
        image = skimage.io.imread(self.paths[index],as_gray=True)
        image = skiform.resize(image, self.size, order=1, preserve_range=True)
        image = normalize(image)
        if(self.gt_path!=None):
            gt_img = skimage.io.imread(self.gt_path[index],as_gray=True)
            gt_mask = skiform.resize(gt_img, self.size, order=0, preserve_range=True)
            gt_mask = gt_mask>0
        else:
            gt_mask = np.zeros_like(image)
        return image,gt_mask
    def __getitem__(self, index):

        image,gt_mask = self.read_image(index)

        #If texture exists have a interpolation image.
        interpolation_choice = np.random.choice(len(self.paths))
        inter_image,_ = self.read_image(interpolation_choice)

        if(self.which_data == 'busi'):
            _mask_img = np.ones(image.shape)


        param_dict = {}

        if(self.return_param):
            image, label, param_dict = self.simulation(image,inter_image, _mask_img,self.num_lesions,gt_mask=gt_mask)

        else:
            image, label = self.simulation(image,inter_image, _mask_img,self.num_lesions,gt_mask=gt_mask)

        return image.astype(np.single),label.astype(np.single),self.paths[index],param_dict

    def __len__(self):
        """Return the dataset size."""
        return len(self.paths)
