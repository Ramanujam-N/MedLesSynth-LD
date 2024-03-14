import numpy as np
from torch.utils.data import ConcatDataset
import nibabel as nib
import json
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter
# from ImageLoader.DatasetCreation import LesionGeneration
from ImageLoader.LesionGeneration2D import LesionGeneration2D
from ImageLoader.LesionGeneration3D import LesionGeneration3D
from ImageLoader.BackgroundGeneration2D import BackgroundGeneration2D
from ImageLoader.BackgroundGeneration3D import BackgroundGeneration3D
from tqdm import tqdm
from datetime import datetime
import argparse
import skimage
import os
def generic_bg_2d(save_path,image_size,data,num_images,phase='train'):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'
    
    num_lesions = np.random.randint(1,3) #5,15
    semi_axes_range = [(5,10),(10,15),(20,25),(100,105)]
    range_sampling = [0.1,0.2,0.3,0.4]
    centroid_scaling = 3
    ellipses = 15


    dataset = ConcatDataset([BackgroundGeneration2D(have_texture=False,have_noise=True,have_edema=False,have_smoothing=True,dark=True,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axes_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    with tqdm(range(num_images)) as pbar:
        for i in pbar:  
            image,label,clean_path,param = dataset[i]

            with open(img_path+ str(i) + '.json', 'w') as f:
                param['clean_path'] = clean_path
                json.dump(param, f)

            skimage.io.imsave(img_path+ str(i)+'_image.png',image)
            skimage.io.imsave(img_path+ str(i)+'_mask.png',label)
            pbar.update(0)

def generic_bg_3d(save_path,image_size,data,num_images,phase='train',status=[0,0,0]):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'

    num_lesions = (1,4)
    semi_axis_range = [(5,10),(10,15),(15,20),(20,25)]
    range_sampling = [0.1,0.2,0.3,0.4]
    centroid_scaling = 15
    ellipses = 15
    dark = True

    dataset = ConcatDataset([BackgroundGeneration3D(have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=True,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    with tqdm(range(status[phase_choice],num_images)) as pbar:
        for i in pbar:  
            image,label,affine,clean_path,param = dataset[i]            

            img = nib.Nifti1Image(image, affine=affine)
            label = nib.Nifti1Image(label, affine=affine)

            with open(img_path+ str(i)+ '.json', 'w') as f:
               param['clean_path'] = clean_path
               json.dump(param, f)

            nib.save(img, img_path+ str(i)+'_FLAIR.nii.gz')
            nib.save(label, img_path+ str(i)+'_manualmask.nii.gz')
            pbar.update(0)

def generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images,phase='train'):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'
    
    if(data=='busi'):
        num_lesions = np.random.randint(1,3) #5,15
        semi_axes_range = [(5,10),(10,15),(20,25),(100,105)]
        range_sampling = [0.1,0.2,0.3,0.4]
        centroid_scaling = 3
        ellipses = 15

    dataset = ConcatDataset([LesionGeneration2D(clean_indexes,have_texture=False,have_noise=True,have_edema=False,have_smoothing=True,dark=True,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axes_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    with tqdm(range(num_images)) as pbar:
        for i in pbar:  
            image,label,clean_path,param = dataset[i]

            with open(img_path+ str(i) + '.json', 'w') as f:
                param['clean_path'] = clean_path
                json.dump(param, f)

            skimage.io.imsave(img_path+ str(i)+'_image.png',image)
            skimage.io.imsave(img_path+ str(i)+'_mask.png',label)
            pbar.update(0)

def generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images,phase='train',status=[0,0,0]):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'

    if(data == 'wmh'):
        num_lesions = (5,30) #5,15
        semi_axis_range = [(2,5),(3,5)]
        range_sampling = [0.8,0.2]
        centroid_scaling = 20
        ellipses = 15
        dark = False
    elif(data == 'brats'):
        num_lesions = (1,4)
        semi_axis_range = [(5,10),(10,15),(15,20),(20,25)]
        range_sampling = [0.1,0.2,0.3,0.4]
        centroid_scaling = 15
        ellipses = 15
        dark = True
    
    print(data,num_lesions)

    dataset = ConcatDataset([LesionGeneration3D(clean_indexes['images'],clean_indexes['masks'],have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=True,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    with tqdm(range(status[phase_choice],num_images)) as pbar:
        for i in pbar:  
            image,label,affine,clean_path,param = dataset[i]            

            img = nib.Nifti1Image(image, affine=affine)
            label = nib.Nifti1Image(label, affine=affine)

            with open(img_path+ str(i)+ '.json', 'w') as f:
               param['clean_path'] = clean_path
               json.dump(param, f)

            nib.save(img, img_path+ str(i)+'_FLAIR.nii.gz')
            nib.save(label, img_path+ str(i)+'_manualmask.nii.gz')
            pbar.update(0)

from ImageLoader.SphereGeneration3D import SphereGeneration
from ImageLoader.RandomShapes3D import RandomShapes3D

def spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images,phase='train',status=[0,0,0]):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'
    
    num_lesions = (1,2)
    semi_axis_range = [(5,10),(10,15)]
    range_sampling = [0.5,0.5,]
    centroid_scaling = 10
    ellipses = 10

    if(data=='spheres'):
        dataset = ConcatDataset([SphereGeneration(clean_indexes['images'],clean_indexes['masks']) for i in range(100)])
    elif(data=='randomshapes'):
        dataset = ConcatDataset([RandomShapes3D(clean_indexes['images'],clean_indexes['masks'],have_texture=False,have_noise=False,have_edema=False,have_smoothing=False,dark=False,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    elif(data=='texshapes'):
        dataset = ConcatDataset([RandomShapes3D(clean_indexes['images'],clean_indexes['masks'],have_texture=True,have_noise=True,have_edema=False,have_smoothing=True,dark=False,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])

    with tqdm(range(status[phase_choice],num_images)) as pbar:
        for i in pbar:  
            image,label,affine, = dataset[i]            

            img = nib.Nifti1Image(image, affine=affine)
            label = nib.Nifti1Image(label, affine=affine)

            nib.save(img, img_path+ str(i)+'_FLAIR.nii.gz')
            nib.save(label, img_path+ str(i)+'_manualmask.nii.gz')
            pbar.update(0)

def create(mode,data,workers,batch,date,on_data,unique_id,status):
    save_path = './simulation_data/'+date+'/'+data
    if(data=='busi'):
        image_size = (598,494)
        image_size = (512,512)

        split_sizes = (305,44,88)
        clean_indexes = glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/BUSI/Dataset_BUSI/Dataset_BUSI_with_GT/normal/*).png')        
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train')
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val')
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test')

    elif(data=='idrid'):
        image_size = (1072,712)
        image_size = (1024,768)

        clean_indexes = glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/Retinal_images/Dataset_types/Normal_STARE/*.png')
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train')
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val')
        generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test')
    elif(data=='wmh'):
        image_size = (132,170,47)
        image_size = (128,128,48)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_2d.npy',allow_pickle=True).item()
        split_sizes = (42,6,12)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='brats'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (876,125,250)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='spheres'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (84,12,24)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='randomshapes'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (84,12,24)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='texshapes'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (84,12,24)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)


if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",default='N',choices=['N','W_gt','W_centroid',],help="Type of simulation")
    parser.add_argument("-data",default='brats',choices=['texshapes','randomshapes','spheres','wmh','brats','busi','lits','stare'],help='Which data to run on?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-status",default=[0,0,0],nargs='+', type=int,)
    parser.add_argument("-on_data",default=False,action='store_true')
    parser.add_argument("-unique_id",default='',help='Name to uniquely identify the experiment')
    args = parser.parse_args()

    print("-----------------------------Arguments for the current simulation creation-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    create(**vars(args))





    """ ****** Include these above *******

            if(self.which_data=='wmh'):
            num_lesions = np.random.randint(1,30) #5,15
            ranges = [(2,5),(3,5)]
            centroid_scaling = 20
        
        elif(self.which_data=='size1'):
            num_lesions = np.random.randint(1,30) #5,15
            ranges = [(2,5),(2,5)]
            centroid_scaling = 20

        elif(self.which_data=='size2'):
            num_lesions = np.random.randint(1,15) #5,15
            ranges = [(3,5),(3,5)]
            centroid_scaling = 20

        elif(self.which_data=='size3'):
            num_lesions = np.random.randint(1,10) #5,15
            ranges = [(5,10),(5,10)]
            centroid_scaling = 15

        elif(self.which_data=='size4'):
            num_lesions = np.random.randint(1,5) #5,15
            ranges = [(10,15),(10,15)]
            centroid_scaling = 15

        elif(self.which_data=='all'):
            num_lesions = np.random.randint(1,30) #5,15
            ranges = [(2,5),(3,5),(5,10),(10,15)]
            centroid_scaling = 20

        elif(self.which_data=='busi'):
            num_lesions = np.random.randint(1,3) #5,15
            ranges = [(5,10),(10,15),(20,25),(100,105)]
            centroid_scaling = 20

        elif(self.which_data=='stare'):
            num_lesions = np.random.randint(1,20) #5,15
            ranges = [(2,5),(3,5),]
            centroid_scaling = 20
        
        else:
            num_lesions = np.random.randint(1,5)
            ranges = [(5,10),(10,15)]
            centroid_scaling = 15
"""