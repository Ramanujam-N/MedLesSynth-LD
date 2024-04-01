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

class BackgroundGenerationLiver3D(Dataset):
    def __init__(self, type_of_imgs='nifty',have_texture=True,have_noise=True, have_smoothing=True, have_small=True, have_edema=True, return_param=True, transform=None, dark=True, which_data='wmh',perturb=False, size=(128,128,128),semi_axis_range=[(5,10)],centroid_scale=10,num_lesions=5,range_sampling=[1],num_ellipses=15):

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
        gaussian_noise = gaussian_filter(noise,sigma) + 0.5*gaussian_filter(noise,sigma/2) + 0.25*gaussian_filter(noise,sigma/4) #+ 0.125*gaussian_filter(noise,sigma/8) + 0.0625*gaussian_filter(noise,sigma/16)
        # gaussian_noise = 1 + 0.1*gaussian_filter(noise,sigma) # 0.4 - 0.6 MR Background
 
        # noise = np.random.random(self.size)
        # gaussian_noise2 = gaussian_filter(noise,sigma) #+ 0.5*gaussian_filter(noise,sigma/4) + 0.25*gaussian_filter(noise,sigma/8) + 0.125*gaussian_filter(noise,sigma/16)
        # gaussian_noise2 -= gaussian_noise2.min()
        # gaussian_noise2 /= gaussian_noise2.max()

        # gaussian_noise2 = ((gaussian_noise2 - gaussian_noise2.min())*(255)/(gaussian_noise.max()-gaussian_noise.min())) + 0

        gaussian_noise_min = gaussian_noise.min()
        gaussian_noise_max = gaussian_noise.max()

        # print(gaussian_noise_min,gaussian_noise_max)
        # Normalizing  (a - amin)*(tmax-tmin)/(a_max - a_min) + tmin
        #tex_noise = ((gaussian_noise - gaussian_noise_min)*(range_max-range_min)/(gaussian_noise_max-gaussian_noise_min)) + range_min 

        tex_noise = gaussian_noise/gaussian_noise_max
        tex_noise*=125

        # gaussian2mask = np.logical_and(gaussian_noise2>125, gaussian_noise2<180)
        # tex_noise[gaussian2mask>0] = gaussian_noise2[gaussian2mask>0]+40
    
        return tex_noise
    
    def shape_generation(self,scale_centroids,centroid_main,num_ellipses,semi_axes_range,perturb_sigma=[1,1,1],image_mask=1):
        # For the number of ellipsoids generate centroids in the local cloud
        # if(self.data == 'brats'):
        random_centroids1 = centroid_main[0].T + (np.random.random((num_ellipses,3)))
        random_centroids2 = centroid_main[1].T + (np.random.random((num_ellipses,3)))


        # Selection of the semi axis length
        # 1. Select the length of the major axis length 
        # 2. Others just need to be a random ratio over the major axis length
        random_major_axes = np.random.randint(semi_axes_range[0],semi_axes_range[1],(num_ellipses,1))
        val1 = np.random.uniform(0.4,0.5,size = (num_ellipses,1))
        val2 = np.random.uniform(0.5,0.6,size = (num_ellipses,1))
        random_minor_axes2 = np.concatenate([val1,0.9*np.ones((num_ellipses,1)),val2],1) #0.1 to 0.8
        random_minor_axes1 = np.concatenate([0.7*np.ones((num_ellipses,1)),val1,val2],1) #0.1 to 0.8

        random_minor_axes_ven = np.concatenate([np.ones((num_ellipses,1)),np.random.uniform(0.4,0.6,size = (num_ellipses,1)),np.random.uniform(0.8,1,size = (num_ellipses,1))],1) #0.1 to 0.8

        random_semi_axes1 = random_major_axes*random_minor_axes1
        random_semi_axes2 = random_major_axes*random_minor_axes2
        random_semi_axes_ven = random_major_axes//3*random_minor_axes_ven

        # Permuting the axes so that one axes doesn't end up being the major every time.
        # rng = np.random.default_rng()
        # random_semi_axes = rng.permuted(random_semi_axes, axis=1)

        
        # Random rotation angles for the ellipsoids
        # random_rot_angles = np.random.uniform(size = (num_ellipses,3))*np.pi
        random_rot_angles1= np.zeros(shape = (num_ellipses,3))
        random_rot_angles2 = np.zeros(shape = (num_ellipses,3))
       
        random_rot_angles2[:,2] +=np.pi/6
        # random_rot_angles2[:,2] *=np.pi
        # random_rot_angles1[:,1] *=(-np.pi/2)

        # random_rot_angles2[:,1] *=np.pi
        # random_rot_angles2[:,2] *=np.pi

#  -------------- make vessels too ----------------
        x,y,z = np.mgrid[0:self.size[0], 0:self.size[1],0:self.size[2]]
        x = np.ones_like(x)*self.size[0]
        n= gaussian_filter(np.random.random(self.size[0]),sigma =8) +0.5*gaussian_filter(np.random.random(self.size),sigma =4) +0.25*gaussian_filter(np.random.random(self.size),sigma =2)
        n -= n.mean()
        n /= n.std()
        folds = (np.sin((x+y+z)/1024 + 2* (n)) + 1) 
        folds -=folds.min()
        folds /=folds.max()+1e-7
        folds = np.logical_and(folds>0.5,folds<0.6)
        folds = ndimage.binary_opening(folds)

        # out_wmh = []
        # for i in range(num_ellipses):
        #     out_wmh.append(self.ellipsoid(random_centroids1[i],*random_semi_axes[i]//1.5,*random_rot_angles[i],img_dim=self.size) )
        
        # out_wmh2 = []
        # for i in range(num_ellipses):
        #     out_wmh2.append(self.ellipsoid(random_centroids2[i],*random_semi_axes[i]//1.5,*random_rot_angles[i],img_dim=self.size) )

        # out_wmh = np.logical_or.reduce(out_wmh)*image_mask
        # out_wmh2 = np.logical_or.reduce(out_wmh2)*image_mask
        # out_wmh = np.logical_or(out_wmh,out_wmh2) 

        # folds = np.logical_or((1-folds)*(1-out_wmh),out_wmh)
        # mask below wale folds ke intensity bada de

        out_strip = []
        for i in range(num_ellipses):
            out_strip.append(self.ellipsoid(random_centroids1[i],*random_semi_axes1[i]//1.1,*random_rot_angles1[i],img_dim=self.size) )
        
        out_strip2 = []
        for i in range(num_ellipses):
            out_strip2.append(self.ellipsoid(random_centroids2[i],*random_semi_axes2[i]//1.1,*random_rot_angles2[i],img_dim=self.size) )

        out_strip = np.logical_or.reduce(out_strip)*image_mask
        out_strip2 = np.logical_or.reduce(out_strip2)*image_mask
        out_strip = np.logical_or(out_strip,out_strip2) 

        out = []
        for i in range(num_ellipses):
            out.append(self.ellipsoid(random_centroids1[i],*random_semi_axes1[i],*random_rot_angles1[i],img_dim=self.size) )
        
        out2 = []
        for i in range(num_ellipses):
            out2.append(self.ellipsoid(random_centroids2[i],*random_semi_axes2[i],*random_rot_angles2[i],img_dim=self.size) )

        out = np.logical_or.reduce(out)*image_mask
        out2 = np.logical_or.reduce(out2)*image_mask
        out = np.logical_or(out,out2) 

        ventricles1 = self.ellipsoid(random_centroids1[i]+[-8,8,0],*random_semi_axes_ven[i],*random_rot_angles1[i]+[0,0,-np.pi*0.9],img_dim=self.size) 
        ventricles2 = self.ellipsoid(random_centroids2[i]+[-8,-8,0],*random_semi_axes_ven[i],*random_rot_angles1[i]+[0,0,+np.pi*0.9],img_dim=self.size) 

        ventricles3 = self.ellipsoid(random_centroids1[i]+[8,8,0],*random_semi_axes_ven[i],*random_rot_angles2[i]+[0,0,+np.pi*0.9],img_dim=self.size) 
        ventricles4 = self.ellipsoid(random_centroids2[i]+[8,-8,0],*random_semi_axes_ven[i],*random_rot_angles2[i]+[0,0,-np.pi*0.9],img_dim=self.size) 
        
        out = out#*(1-ventricles1)*(1-ventricles2)*(1-ventricles3)*(1-ventricles4)
        ventricles = 1- (1-ventricles1)*(1-ventricles2)*(1-ventricles3)*(1-ventricles4)
        folds = (1-(1-folds)*out)*out
        out_whole = out
        #out = out*folds
        regions = skimage.measure.regionprops(np.int16(out*2))

        if(regions==[]):
            return np.zeros_like(out),-1

        return out,out_whole,out_strip,ventricles,folds,1


    def simulation(self,image_mask):
        param_dict = {}
        roi = np.zeros_like(image_mask)
        roi[self.size[0]//4:(self.size[0]*3)//4,self.size[1]//4:(self.size[1]*3)//4,self.size[2]//4:(self.size[2]*3)//4] = 1
        roi_with_masks = roi
        output_image = np.zeros_like(image_mask) 
        output_mask = np.zeros_like(image_mask)
        
        total_param_list = ['scale_centroid','num_ellipses','semi_axes_range','alpha','beta','gamma','smoothing_mask',
                            'tex_sigma','range_min','range_max','tex_sigma_edema','pertub_sigma']
        

        num_lesions = 1
        gamma = 0
        tex_sigma_edema = 0

        # x_corr,y_corr,z_corr = np.nonzero(roi_with_masks[:,:,:])
        # random_coord_index = np.random.choice(len(x_corr),1)
        # centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
        centroid_main = (np.array([64,64,64]),np.array([80,48,64]))
        # print(centroid_main,roi_with_masks[centroid_main[0],centroid_main[1],centroid_main[2]])
        # We need a loop and random choices tailored here for multiple lesions 

        scale_centroid = np.random.randint(2,self.centroid_scaling)
        num_ellipses = 1 #np.random.randint(5,self.num_ellipses)
        semi_axes_range = self.ranges[int(np.random.choice(len(self.ranges),p=self.range_sampling))]

            
        if(semi_axes_range==(2,5)):
            alpha = np.random.uniform(0.5,0.8)
            beta = 1-alpha
        if(semi_axes_range!=(2,5)):
            alpha = np.random.uniform(0.6,0.8)
            beta = 1-alpha


        smoothing_mask = np.random.uniform(0.7,0.9)
        smoothing_image = np.random.uniform(0.3,0.5)

        # tex_sigma = np.random.uniform(0.6,0.8)  # old 1-4
        tex_sigma = np.random.uniform(1,2)  # MRI 0.4-0.6

        range_min = 0 #np.random.uniform(-0.5,0.5)
        range_max = 1 #np.random.uniform(0.7,1)
        perturb_sigma = np.random.uniform(0.5,5)

        #print(alpha,beta)
        if(semi_axes_range == (2,5)):
            small_sigma = np.random.uniform(2,3)
            out = self.gaussian_small_shapes(image_mask,small_sigma)
        else:   
            out,out_2,out_strip,ventricles,folds,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask) 
            while(shape_status==-1):
                print(shape_status)
                # random_coord_index = np.random.choice(len(x_corr),1)
                # centroid_main = np.array([x_corr[random_coord_index],y_corr[random_coord_index],z_corr[random_coord_index]])
                centroid_main = (np.array([64,32,64]),np.array([32,64,64]))
                out,shape_status = self.shape_generation(scale_centroid, centroid_main,num_ellipses,semi_axes_range,perturb_sigma,image_mask=image_mask)
        
        output_mask = np.logical_or(output_mask,out)

        if(self.have_noise and semi_axes_range !=(2,5)):
            tex_noise = self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
        else:
            tex_noise = 1.0
        
        output_image = out*tex_noise*(1-folds) + 150*folds#- 0.2*out_strip*out*self.gaussian_noise(tex_sigma,self.size,range_min,range_max)
        output_image -=output_image.min()
        output_image /=output_image.max()

        output_image *=255
        output_image =output_image.astype(np.int16)
        # output_image = skimage.morphology.opening(output_image,np.array([[[0,0,0],
        #                                                  [0,0,0],
        #                                                  [1,0,0]],[[0,0,0],
        #                                                  [0,0,0],
        #                                                  [0,1,0]],[[0,0,1],
        #                                                  [0,0,0],
        #                                                  [0,0,0]]]))
        # output_mask = skimage.morphology.binary_dilation(output_mask,skimage.morphology.cube(2))
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
        nii_img_affine = np.eye(4)

        brain_mask_img = np.ones(self.size)
        param_dict = {}

        if(self.return_param):
            image, label, param_dict = self.simulation(brain_mask_img,)
        else:
            image, label = self.simulation(brain_mask_img,)

        return image.astype(np.single),label.astype(np.single),nii_img_affine,param_dict

    def __len__(self):
        """Return the dataset size."""
        return 100

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
