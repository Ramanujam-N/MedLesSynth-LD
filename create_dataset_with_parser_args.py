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
from ImageLoader.BackgroundGenerationLiver3D import BackgroundGenerationLiver3D

from tqdm import tqdm
from datetime import datetime
import argparse
import skimage
from helper_datadict  import helper_path_configuration
import os

def sim_indexes(date,data):
    return {'images':glob.glob('./simulation_data/'+date+'/background3d'+data+'/TrainSet/*FLAIR*'),
    'masks':glob.glob('./simulation_data/'+date+'/background3d'+data+'/TrainSet/*manualmask*')
    }


def generic_bg_2d(save_path,image_size,data,num_images,phase='train',status=[0,0,0]):
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
    with tqdm(range(status[phase_choice],num_images)) as pbar:
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
    semi_axis_range = [(55,60)]
    range_sampling = [1]
    centroid_scaling = 20
    ellipses = 15
    dark = False

    if(data=='background3dliver'):
        dataset = ConcatDataset([BackgroundGenerationLiver3D(have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    else:
        dataset = ConcatDataset([BackgroundGeneration3D(have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    with tqdm(range(status[phase_choice],num_images)) as pbar:
        for i in pbar:  
            image,label,affine,param_dict = dataset[i]            

            img = nib.Nifti1Image(image, affine=affine)
            label = nib.Nifti1Image(label, affine=affine)

            nib.save(img, img_path+ str(i)+'_FLAIR.nii.gz')
            nib.save(label, img_path+ str(i)+'_manualmask.nii.gz')
            pbar.update(0)

def generate_and_save_2d(clean_indexes,save_path,image_size,data,num_images,phase='train',status=[0,0,0],clean=True):
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


    if(clean==True):
        dataset = ConcatDataset([LesionGeneration2D(img_path=clean_indexes,gt_path=None,have_texture=False,have_noise=True,have_edema=False,have_smoothing=True,dark=True,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axes_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    else:
        dataset = ConcatDataset([LesionGeneration2D(img_path=clean_indexes['train_names_flair'],gt_path=clean_indexes['train_names_seg'],have_texture=False,have_noise=True,have_edema=False,have_smoothing=True,dark=True,which_data=data,size=image_size,perturb=False,num_lesions=num_lesions,semi_axis_range=semi_axes_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])

    with tqdm(range(status[phase_choice],num_images)) as pbar:
        for i in pbar:  
            image,label,clean_path,param = dataset[i]

            with open(img_path+ str(i) + '.json', 'w') as f:
                param['clean_path'] = clean_path
                json.dump(param, f)

            skimage.io.imsave(img_path+ str(i)+'_image.png',image)
            skimage.io.imsave(img_path+ str(i)+'_mask.png',label)
            pbar.update(0)

def generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images,phase='train',status=[0,0,0],clean=True):
    phase_choice = {'train':0,'val':1,'test':2}[phase]
    img_type = ['Training','Validation','Testing']
    img_folder = ['TrainSet','ValSet','TestSet']
    img_path = save_path+'/'+img_folder[phase_choice]+'/' 
    os.makedirs(img_path,exist_ok=True)
    img_path+=img_type[phase_choice]+'_'

    if(data == 'wmh'):
        num_lesions = (5,30) #5,15
        if(clean==False):
            num_lesions = (3,15)
        semi_axis_range = [(2,5),(3,5)]
        range_sampling = [0.8,0.2]
        centroid_scaling = 20
        ellipses = 15
        dark = False
        perturb = True

    elif(data == 'brats'):
        num_lesions = (1,4)
        if(clean==False):
            num_lesions = (1,3)
        semi_axis_range = [(5,10),(10,15),(15,20),(20,25)]
        range_sampling = [0.1,0.2,0.3,0.4]
        centroid_scaling = 15
        ellipses = 15
        dark = True
        perturb = True
    
    elif(data == 'lits'):
        num_lesions = (1,4) #5,15
        semi_axis_range = [(3,5),(5,10)]
        range_sampling = [0.3,0.7]
        centroid_scaling = 15
        ellipses = 15
        dark = True
        perturb = False    
    
    print(data,num_lesions)

    if(clean==True):
        dataset = ConcatDataset([LesionGeneration3D(img_path=clean_indexes['images'],gt_path=None,mask_path=clean_indexes['masks'],have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=True,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])
    else:
        dataset = ConcatDataset([LesionGeneration3D(img_path=clean_indexes['train_names_flair'],gt_path=clean_indexes['train_names_seg'],mask_path=None,have_texture=True,have_noise=True,have_edema=True,have_smoothing=True,dark=dark,which_data=data,size=image_size,perturb=perturb,num_lesions=num_lesions,semi_axis_range=semi_axis_range,centroid_scale= centroid_scaling,range_sampling=range_sampling,num_ellipses=ellipses) for i in range(100)])

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

def create(mode,data,workers,batch,date,on_data,unique_id,status,system_data_path):
    save_path = './simulation_data/'+date+'/'+data

#---------------------------------------------------- To Generate on Simulated Lesions------------------------------------------------------------
    
    if(data=='busi'):
        image_size = (598,494)
        image_size = (512,512)

        split_sizes = (42*5,6*5,12*5)
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
        split_sizes = (42*5,6*5,12*5)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='brats'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (42*5,6*5,12*5)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

    elif(data=='lits'):
        image_size = (137,172,139)
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_clean_liver.npy',allow_pickle=True).item()
        split_sizes = (42*5,6*5,12*5)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
#---------------------------------------------------- To Generate on Random Shapes (Ablation) ------------------------------------------------------------

    elif(data=='spheres'):
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (42*5,6*5,12*5)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='randomshapes'):
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (42*5,6*5,12*5)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)
    elif(data=='texshapes'):
        image_size = (128,128,128)

        clean_indexes = np.load('./Data_splits/Clean_indexes/complete_nimh_3d.npy',allow_pickle=True).item()
        split_sizes = (42*5,6*5,12*5)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        spheres_generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

#---------------------------------------------------- Generic BG Generation------------------------------------------------------------
        
    elif(data=='background3d' or data=='background3dliver'):
        image_size = (128,128,128)

        split_sizes = (42*5,6*5,12*5)
        generic_bg_3d(save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generic_bg_3d(save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generic_bg_3d(save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

#---------------------------------------------------- To Generate on Generic BG------------------------------------------------------------
        
    elif(data=='bratsonsim'):
        image_size = (128,128,128)

        clean_indexes = sim_indexes(date,'')
        split_sizes = (42*5,6*5,12*5)
        data='brats'
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

    elif(data=='wmhonsim'):
        image_size = (128,128,128)

        clean_indexes = sim_indexes(date,'')
        split_sizes = (42*5,6*5,12*5)
        data='wmh'
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

    elif(data=='litsonsim'):
        image_size = (128,128,128)

        clean_indexes = sim_indexes(date,'liver')
        split_sizes = (42*5,6*5,12*5)
        data='lits'
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status)
        generate_and_save_3d(clean_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status)

#---------------------------------------------------- To Generate on Train Data ------------------------------------------------------------
        
    elif(data=='bratsonbrats'):
        image_size = (128,128,128)
        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()
        brats_indexes = helper_path_configuration(brats_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='brats'
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)

    elif(data=='wmhonbrats'):
        image_size = (128,128,128)
        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()
        brats_indexes = helper_path_configuration(brats_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='wmh'
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_3d(brats_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)


    elif(data=='bratsonwmh'):
        image_size = (128,128,128)
        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()
        wmh_indexes = helper_path_configuration(wmh_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='brats'
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)


    elif(data=='wmhonwmh'):
        image_size = (128,128,128)
        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()
        wmh_indexes = helper_path_configuration(wmh_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='wmh'
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_3d(wmh_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)

    elif(data=='litsonlits'):
        image_size = (128,128,128)
        lits_indexes = np.load('./Data_splits/Train_val_test_42_6_x/liver_indexes.npy', allow_pickle=True).item()
        lits_indexes = helper_path_configuration(lits_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='lits'
        generate_and_save_3d(lits_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_3d(lits_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_3d(lits_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)

    elif(data=='busionbusi'):
        image_size = (512,512)
        busi_indexes = np.load('./Data_splits/Train_val_test_42_6_x/busi_indexes.npy', allow_pickle=True).item()
        busi_indexes = helper_path_configuration(busi_indexes,system_data_path)

        split_sizes = (42*5,6*5,12*5)
        data='busi'
        generate_and_save_2d(busi_indexes,save_path,image_size,data,num_images=split_sizes[0],phase='train',status=status,clean=False)
        generate_and_save_2d(busi_indexes,save_path,image_size,data,num_images=split_sizes[1],phase='val',status=status,clean=False)
        generate_and_save_2d(busi_indexes,save_path,image_size,data,num_images=split_sizes[2],phase='test',status=status,clean=False)


if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",default='N',choices=['N','W_gt','W_centroid',],help="Type of simulation")
    parser.add_argument("-data",default='brats',choices=['busionbusi','litsonlits','bratsonwmh','wmhonwmh','wmhonbrats','bratsonbrats','litsonsim','wmhonsim','bratsonsim','background2d','background3d','background3dliver','texshapes','randomshapes','spheres','wmh','brats','busi','lits','stare'],help='Which data to run on?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-status",default=[0,0,0],nargs='+', type=int,)
    parser.add_argument("-on_data",default=False,action='store_true')
    parser.add_argument("-unique_id",default='',help='Name to uniquely identify the experiment')
    parser.add_argument("-data_path",dest='system_data_path',default=112,type=int,choices=[112,63,64])
    args = parser.parse_args()

    data_addresses = {63:'/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6',64:'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a',112:'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f',}
    args.system_data_path = data_addresses[args.system_data_path]

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