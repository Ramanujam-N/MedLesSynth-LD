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

class BackgroundGeneration2D(Dataset):
    def __init__(self,type_of_imgs='png',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15,num_imgs=210):
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
        self.num_imgs = num_imgs
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
        # noise = np.random.random(size)
        # gaussian_noise = gaussian_filter(noise,sigma) + 0.5*gaussian_filter(noise,sigma/2) + 0.25*gaussian_filter(noise,sigma/4)
        # gaussian_noise_min = gaussian_noise.min()
        # gaussian_noise_max = gaussian_noise.max()

        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        # tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 


        x,y = np.mgrid[0:size[0], 0:size[1]]


        n= gaussian_filter(np.random.randn(*size),sigma =8) +0.5*gaussian_filter(np.random.randn(*size),sigma =4) +0.25*gaussian_filter(np.random.randn(*size),sigma =2)
        n -= n.mean()
        n /= n.std()
        out = (np.sin(-(x/8) + 1.5* (n)) + 1)  
        out -=out.min()
        out /=out.max()+1e-7
        ss = (out+n+0.5*np.random.randn(*size))
        ss -=ss.min()
        ss /= ss.max()
        ss = ss*127

        ss = (ss - x)
        ss -=ss.min()
        ss /= ss.max()
        ss = ss**2.5

        tex_noise = ss
        return tex_noise
    
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1],image_mask=1):
        # For the number of ellipsoids generate centroids in the local cloud
        random_centroids = centroid_main.T + (np.random.random((num_ellipses,2))*scale_centroids)


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = np.random.randint(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        random_minor_axes = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.1,0.8,size = (num_ellipses,1))],1)
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

    def simulation(self):
        param_dict = {}
                
        total_param_list = ['tex_sigma','range_min','range_max']
        

        tex_sigma = np.random.uniform(0.4,0.7)
        range_min = np.random.uniform(-0.5,0.5)
        range_max = np.random.uniform(0.7,1)

        tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
        
        output_image = tex_noise
        output_image -= output_image.min()
        output_image /= output_image.max()

        # roi_with_masks *= (1-output_mask)>0
        
        total_params = [tex_sigma,range_min,range_max]
        
        for j in range(len(total_params)):
            param_dict[total_param_list[j]] = total_params[j]

        output_mask = np.ones(self.size)
        if(self.return_param):
            return output_image, output_mask, param_dict
        else:
            return output_image, output_mask

    def read_image(self,index):
        image = skimage.io.imread(self.paths[index],as_gray=True)
        image = skiform.resize(image, self.size, order=1, preserve_range=True)
        image = normalize(image)
        return image
    
    def __getitem__(self, index):

        param_dict = {}
        
        if(self.return_param):
            image, label, param_dict = self.simulation()

        else:
            image, label = self.simulation()

        return image.astype(np.single),label.astype(np.single),param_dict

    def __len__(self):
        """Return the dataset size."""
        return self.num_imgs
