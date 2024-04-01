from ModelArchitecture.Transformations import *
from ImageLoader.ImageLoader3D import ImageLoader3D
from ImageLoader.ImageLoader2D import ImageLoader2D
from ImageLoader.FPILoader3D import FPILoader3D
from ModelArchitecture.DUCK_Net import DuckNet,DuckNet_smaller
from ModelArchitecture.UNet import NestedUNet,UNet,HalfUNet,ResUNet,SA_UNet,SAC_UNet
from ModelArchitecture.UNet2D import NestedUNet2D,UNet2D,HalfUNet2D,ResUNet2D,SA_UNet2D,SAC_UNet2D
from ModelArchitecture.UNETR import VITForSegmentation
from ModelArchitecture.UNETR2D import VITForSegmentation2D
from ModelArchitecture.Losses import WBCE_DICELoss,WBCE_FOCALDICELoss,FocalLoss,WBCE_FOCALLoss,FOCAL_DICELoss
from ModelArchitecture.Losses import LogCoshDiceLoss,BCE_Loss_Weighted,MS_SSIMLoss,Frequency_loss
from ModelArchitecture.Losses_truenet import CombinedLoss,DiceLoss
from ModelArchitecture.Losses_AdaptiveRegionSpecific import Adaptive_Region_Specific_TverskyLoss
from ModelArchitecture.metrics import Dice_Score
import matplotlib.pyplot as plt
import sys
import os
import nibabel as nib
import torch.nn as nn
import glob
import numpy as np
from tqdm import tqdm
import skimage
import torchvision
from torch.utils.data import DataLoader,ConcatDataset

CURRENT_DIRECTORY = os.getcwd()
PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)

sys.path.insert(1, PARENT_DIRECTORY+'/Slim-UNETR')
from src.SlimUNETR.SlimUNETR import SlimUNETR
from src.SlimUNETR2D.SlimUNETR2D import SlimUNETR2D


composed_transform = transforms.Compose([
            RandomRotation3D([-10,10],p=0.15),
            RandomHorizontalFlip3D(p=0.15),
            RandomNoise3D(p=0.15),
            #RandomBrightness3D(),
            ToTensor3D(True)])

composed_transform_2d = transforms.Compose([
            RandomRotation2D([-10,10],p=0.15),
            RandomHorizontalFlip2D(p=0.15),
            RandomNoise2D(p=0.15),
            #RandomBrightness2D(),
            ToTensor2D(True)])

def helper_resize(image,output,label,shape=(),crop_para=[]):
    image=image.cpu().numpy().squeeze()
    output=output.cpu().numpy().squeeze()
    label=label.cpu().numpy().squeeze()
    if(len(label.shape)<=2):
        return skimage.transform.resize(image, output_shape = label.shape,order=1,preserve_range=True),skimage.transform.resize(output, output_shape = label.shape,order=0,preserve_range=True),label
    
    shape = (int(shape[0]),int(shape[1]),int(shape[2]))
    #print(shape,label.shape)
    image = skimage.transform.resize(image, output_shape = shape,order=1,preserve_range=True)
    output = skimage.transform.resize(output, output_shape = shape,order=0,preserve_range=True)
    actual_image = np.zeros_like(label)
    actual_output = np.zeros_like(label)
    actual_image[crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = image
    actual_output[crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = output
    
    return actual_image,actual_output,label


def helper_transformation(no_aug):
    if(no_aug):
        global composed_transform
        composed_transform = ToTensor3D(True)

        global composed_transform_2d
        composed_transform_2d = ToTensor2D(True)

def helper_path_configuration(indexes,data_path):
    new_header_string = data_path
    z = {}
    names_list = list(indexes.keys())
    for name  in names_list:
        z[name] = [new_header_string + indexes[name][i][41:] for i in range(len(indexes[name]))]
    return z


def helper_supervised(system_data_path,which_data='brats',size=(128,128,128),size2=(128,128,48),no_crop=False,combined_brats_samples=250):
    if(which_data=='wmh'):
        # White Matter Hyperintensities
        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()
        wmh_indexes = helper_path_configuration(wmh_indexes,system_data_path)

        datadict_train = ImageLoader3D(wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'],image_size=size,type_of_imgs='nifty',transform = composed_transform,data='wmh',no_crop=no_crop)
        datadict_val = ImageLoader3D(wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),data='wmh',no_crop=no_crop)
        datadict_test = ImageLoader3D(wmh_indexes['test_names_flair'],wmh_indexes['test_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),data='wmh',no_crop=no_crop,return_size=True,return_orig=True)

    elif(which_data=='brats'):
        # Brain Tumor Segmentation Challenge
        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()
        brats_indexes = helper_path_configuration(brats_indexes,system_data_path)

        datadict_train = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],image_size=size,type_of_imgs='nifty',transform = composed_transform,data='brats',no_crop=no_crop)
        datadict_val = ImageLoader3D(brats_indexes['val_names_flair'],brats_indexes['val_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),data='brats',no_crop=no_crop)
        datadict_test = ImageLoader3D(brats_indexes['test_names_flair'],brats_indexes['test_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),data='brats',no_crop=no_crop,return_size=True,return_orig=True)

    elif(which_data=='lits'):
        # Liver Tumor Segmentation
        liver_indexes = np.load('./Data_splits/Train_val_test_42_6_x/liver_indexes.npy', allow_pickle=True).item()
        liver_indexes = helper_path_configuration(liver_indexes,system_data_path)

        datadict_train = ImageLoader3D(liver_indexes['train_names_flair'],liver_indexes['train_names_seg'],type_of_imgs='nifty',transform = composed_transform,data='liver')
        datadict_val = ImageLoader3D(liver_indexes['val_names_flair'],liver_indexes['val_names_seg'],type_of_imgs='nifty', transform = ToTensor3D(True),data='liver')
        datadict_test = ImageLoader3D(liver_indexes['test_names_flair'],liver_indexes['test_names_seg'],type_of_imgs='nifty', transform = ToTensor3D(True),data='liver',no_crop=no_crop,return_size=True,return_orig=True)


    elif(which_data=='busi'):
        # Brain Tumor Segmentation Challenge
        busi_indexes = np.load('./Data_splits/Train_val_test_42_6_x/busi_indexes.npy', allow_pickle=True).item()
        busi_indexes = helper_path_configuration(busi_indexes,system_data_path)

        datadict_train = ImageLoader2D(busi_indexes['train_names_flair'],busi_indexes['train_names_seg'],image_size=size,type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(busi_indexes['val_names_flair'],busi_indexes['val_names_seg'],image_size=size,type_of_imgs='png', transform = ToTensor2D(True),data='busi')
        datadict_test = ImageLoader2D(busi_indexes['test_names_flair'],busi_indexes['test_names_seg'],image_size=size,type_of_imgs='png', transform = ToTensor2D(True),data='busi',return_size=True,return_orig=True)

    elif(which_data=='idrid'):
        # Brain Tumor Segmentation Challenge
        idrid_indexes = np.load('./Data_splits/Train_val_test_42_6_x/idrid_indexes.npy', allow_pickle=True).item()
        idrid_indexes = helper_path_configuration(idrid_indexes,system_data_path)

        datadict_train = ImageLoader2D(idrid_indexes['train_names_flair'],idrid_indexes['train_names_seg'],image_size=512,type_of_imgs='png',transform = composed_transform_2d,data='idrid')
        datadict_val = ImageLoader2D(idrid_indexes['val_names_flair'],idrid_indexes['val_names_seg'],image_size=512,type_of_imgs='png', transform = ToTensor2D(True),data='idrid')
        datadict_test = ImageLoader2D(idrid_indexes['test_names_flair'],idrid_indexes['test_names_seg'],image_size=512,type_of_imgs='png', transform = ToTensor2D(True),data='idrid',return_size=True,return_orig=True)
    
    print(len(datadict_train),len(datadict_val),len(datadict_test))
    return datadict_train,datadict_val,datadict_test

def helper_pre_training(which_data='brats',scale_factor=1.0,sim_path_other=None,size=(128,128,128),no_crop=False):
    sim_path = sim_path_other
    print('Using simulation path:{} for data {}'.format(sim_path_other,which_data))


    if(which_data=='wmh'):
        # WMH

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz'))) 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty',image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  
        print(len(train_names_flair),len(train_names_seg))

    elif(which_data=='brats'):
        # BraTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz'))) 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  

    elif(which_data=='lits'):
        # BraTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  

    elif(which_data=='total3d'):
        # BraTS
        train_names_flair = []
        train_names_seg = []
        val_names_flair = []
        val_names_seg = []
        for which_data in ['bratsonsim','wmhonsim','litsonsim']:
            train_names_flair += sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz'))) 
            train_names_seg += sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))

            val_names_flair += sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))
            val_names_seg += sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))
        
        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  

    elif(which_data=='busi'):

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*image.png')))
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*mask.png')))

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*image.png')))
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*mask.png')))
        
        datadict_train = ImageLoader2D(train_names_flair,train_names_seg,image_size=(512,512),type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(val_names_flair,val_names_seg,image_size=(512,512),type_of_imgs='png', transform = ToTensor2D(True),data='busi')

    return datadict_train,datadict_val


def helper_self_supervised(which_data='brats',scale_factor=1.0,sim_path_other=None,size=(128,128,128),no_crop=False):
    sim_path = sim_path_other
    print('Using simulation path:{} for data {}'.format(sim_path_other,which_data))
    train_size = np.ceil(42*scale_factor).astype(np.int16)
    val_size = np.ceil(6*scale_factor).astype(np.int16)

    if(which_data=='wmh'):
        # WMH

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz'))) [:val_size]

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty',image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  
        print(len(train_names_flair),len(train_names_seg))

    elif(which_data=='brats'):
        # BraTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))[:val_size]

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  

    elif(which_data=='lits'):
        # BraTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))[:val_size]

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', image_size=size, transform = composed_transform,no_crop=no_crop)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', image_size=size, transform = ToTensor3D(True),no_crop=no_crop)  

    elif(which_data=='busi'):

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*image.png')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*mask.png')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*image.png')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*mask.png')))[:val_size]
        
        datadict_train = ImageLoader2D(train_names_flair,train_names_seg,image_size=(512,512),type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(val_names_flair,val_names_seg,image_size=(512,512),type_of_imgs='png', transform = ToTensor2D(True),data='busi')

    elif(which_data=='idrid'):

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*image.png')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*mask.png')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*image.png')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*mask.png')))[:val_size]
        
        datadict_train = ImageLoader2D(train_names_flair,train_names_seg,image_size=(512,512),type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(val_names_flair,val_names_seg,image_size=(512,512),type_of_imgs='png', transform = ToTensor2D(True),data='busi')

    return datadict_train,datadict_val



def helper_data_augmentation(system_data_path,which_data='brats',factor=1.0,scale_factor=1.0,sim_path_other=None,size=(128,128,128),no_crop=False):
    sim_path = sim_path_other
    print('Using simulation path:{} for data {}'.format(sim_path_other,which_data))
    augementation_factor = factor

    
    train_size = 42
    val_size = 6

    if(which_data=='wmh'):
        # White Matter Hyperintensities
        
        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()
        wmh_indexes = helper_path_configuration(wmh_indexes,system_data_path)

        wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'] = wmh_indexes['train_names_flair'][:np.int16(np.ceil(augementation_factor*train_size))],wmh_indexes['train_names_seg'][:np.int16(np.ceil(augementation_factor*train_size))] 
        wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'] = wmh_indexes['val_names_flair'][:np.int16(np.ceil(augementation_factor*val_size))],wmh_indexes['val_names_seg'][:np.int16(np.ceil(augementation_factor*val_size))] 


        real_datadict_train = ImageLoader3D(wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'],image_size=size,type_of_imgs='nifty',transform = composed_transform,no_crop=no_crop)
        real_datadict_val = ImageLoader3D(wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),no_crop=no_crop)

    elif(which_data=='brats'):
        # BraTS

        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()
        brats_indexes = helper_path_configuration(brats_indexes,system_data_path)

        brats_indexes['train_names_flair'],brats_indexes['train_names_seg'] = brats_indexes['train_names_flair'][:np.int16(np.ceil(augementation_factor*train_size))],brats_indexes['train_names_seg'][:np.int16(np.ceil(augementation_factor*train_size))] 
        brats_indexes['val_names_flair'],brats_indexes['val_names_seg'] = brats_indexes['val_names_flair'][:np.int16(np.ceil(augementation_factor*val_size))],brats_indexes['val_names_seg'][:np.int16(np.ceil(augementation_factor*val_size))] 

        real_datadict_train = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],image_size=size,type_of_imgs='nifty',transform = composed_transform,no_crop=no_crop)
        real_datadict_val = ImageLoader3D(brats_indexes['val_names_flair'],brats_indexes['val_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),no_crop=no_crop)
    
    elif(which_data=='lits'):
        # LiTS

        liver_indexes = np.load('./Data_splits/Train_val_test_42_6_x/liver_indexes.npy', allow_pickle=True).item()
        liver_indexes = helper_path_configuration(liver_indexes,system_data_path)

        liver_indexes['train_names_flair'],liver_indexes['train_names_seg'] = liver_indexes['train_names_flair'][:np.int16(np.ceil(augementation_factor*train_size))],liver_indexes['train_names_seg'][:np.int16(np.ceil(augementation_factor*train_size))] 
        liver_indexes['val_names_flair'],liver_indexes['val_names_seg'] = liver_indexes['val_names_flair'][:np.int16(np.ceil(augementation_factor*val_size))],liver_indexes['val_names_seg'][:np.int16(np.ceil(augementation_factor*val_size))] 

        real_datadict_train = ImageLoader3D(liver_indexes['train_names_flair'],liver_indexes['train_names_seg'],image_size=size,type_of_imgs='nifty',transform = composed_transform,no_crop=no_crop,data='liver')
        real_datadict_val = ImageLoader3D(liver_indexes['val_names_flair'],liver_indexes['val_names_seg'],image_size=size,type_of_imgs='nifty', transform = ToTensor3D(True),no_crop=no_crop,data='liver')

    elif(which_data=='busi'):
        busi_indexes = np.load('./Data_splits/Train_val_test_42_6_x/busi_indexes.npy', allow_pickle=True).item()
        busi_indexes = helper_path_configuration(busi_indexes,system_data_path)

        real_datadict_train = ImageLoader2D(busi_indexes['train_names_flair'],busi_indexes['train_names_seg'],image_size=(512,512),type_of_imgs='nifty',transform = composed_transform_2d,data='busi')
        real_datadict_val = ImageLoader2D(busi_indexes['val_names_flair'],busi_indexes['val_names_seg'],image_size=(512,512),type_of_imgs='nifty', transform = ToTensor2D(True),data='busi')
    elif(which_data=='idrid'):
        idrid_indexes = np.load('./Data_splits/Train_val_test_42_6_x/idrid_indexes.npy', allow_pickle=True).item()
        idrid_indexes = helper_path_configuration(idrid_indexes,system_data_path)

        real_datadict_train = ImageLoader2D(idrid_indexes['train_names_flair'],idrid_indexes['train_names_seg'],image_size=(512,512),type_of_imgs='nifty',transform = composed_transform_2d,data='busi')
        real_datadict_val = ImageLoader2D(idrid_indexes['val_names_flair'],idrid_indexes['val_names_seg'],image_size=(512,512),type_of_imgs='nifty', transform = ToTensor2D(True),data='busi')


    train_size = np.ceil(42*scale_factor).astype(np.int16)
    val_size = np.ceil(6*scale_factor).astype(np.int16)

    if(which_data=='wmh'):
        # WMH
        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))[:val_size]

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = composed_transform)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = ToTensor3D(True))  

    elif(which_data=='brats'):
        # BraTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))[:val_size]

        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = composed_transform)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = ToTensor3D(True))  

    elif(which_data=='lits'):
        # LiTS

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*FLAIR.nii.gz')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*manualmask.nii.gz')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*FLAIR.nii.gz')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*manualmask.nii.gz')))[:val_size]


        datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = composed_transform)   
        datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty',image_size=size,no_crop=no_crop, transform = ToTensor3D(True))  

    elif(which_data=='busi'):

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*image.png')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*mask.png')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*image.png')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*mask.png')))[:val_size]

        
        datadict_train = ImageLoader2D(train_names_flair,train_names_seg,image_size=(512,512),type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(val_names_flair,val_names_seg,image_size=(512,512),type_of_imgs='png', transform = ToTensor2D(True),data='busi')

    elif(which_data=='idrid'):

        train_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*image.png')))[:train_size] 
        train_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/TrainSet/*mask.png')))[:train_size]

        val_names_flair = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*image.png')))[:val_size]
        val_names_seg = sorted(glob.glob((sim_path+'/'+which_data+'/ValSet/*mask.png')))[:val_size]

        
        datadict_train = ImageLoader2D(train_names_flair,train_names_seg,image_size=(512,512),type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(val_names_flair,val_names_seg,image_size=(512,512),type_of_imgs='png', transform = ToTensor2D(True),data='busi')


    datadict_train = torch.utils.data.ConcatDataset([real_datadict_train,datadict_train])
    datadict_val = torch.utils.data.ConcatDataset([real_datadict_val,datadict_val])
    return datadict_train,datadict_val


def helper_fine_tuning(system_data_path,which_data='brats',factor=1.0,scale_factor=1.0,size=(128,128,128),no_crop=False):
    fine_tuning_factor = factor
    train_size = 42
    val_size = 6

    if(which_data=='wmh'):
        # White Matter Hyperintensities

        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()
        wmh_indexes = helper_path_configuration(wmh_indexes,system_data_path)

        wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'] = wmh_indexes['train_names_flair'][:np.int16(np.ceil(fine_tuning_factor*train_size))],wmh_indexes['train_names_seg'][:np.int16(np.ceil(fine_tuning_factor*train_size))] 
        wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'] = wmh_indexes['val_names_flair'][:np.int16(np.ceil(fine_tuning_factor*val_size))],wmh_indexes['val_names_seg'][:np.int16(np.ceil(fine_tuning_factor*val_size))] 


        datadict_train = ImageLoader3D(wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty',transform = composed_transform)
        datadict_val = ImageLoader3D(wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty', transform = ToTensor3D(True))

    elif(which_data=='brats'):
        # Brain Tumor Segmentation Challenge


        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()
        brats_indexes = helper_path_configuration(brats_indexes,system_data_path)

        brats_indexes['train_names_flair'],brats_indexes['train_names_seg'] = brats_indexes['train_names_flair'][:np.int16(np.ceil(fine_tuning_factor*train_size))],brats_indexes['train_names_seg'][:np.int16(np.ceil(fine_tuning_factor*train_size))] 
        brats_indexes['val_names_flair'],brats_indexes['val_names_seg'] = brats_indexes['val_names_flair'][:np.int16(np.ceil(fine_tuning_factor*val_size))],brats_indexes['val_names_seg'][:np.int16(np.ceil(fine_tuning_factor*val_size))] 

        
        datadict_train = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty',transform = composed_transform)
        datadict_val = ImageLoader3D(brats_indexes['val_names_flair'],brats_indexes['val_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty', transform = ToTensor3D(True))

    elif(which_data=='lits'):
        # Brain Tumor Segmentation Challenge


        liver_indexes = np.load('./Data_splits/Train_val_test_42_6_x/liver_indexes.npy', allow_pickle=True).item()
        liver_indexes = helper_path_configuration(liver_indexes,system_data_path)

        liver_indexes['train_names_flair'],liver_indexes['train_names_seg'] = liver_indexes['train_names_flair'][:np.int16(np.ceil(fine_tuning_factor*train_size))],liver_indexes['train_names_seg'][:np.int16(np.ceil(fine_tuning_factor*train_size))] 
        liver_indexes['val_names_flair'],liver_indexes['val_names_seg'] = liver_indexes['val_names_flair'][:np.int16(np.ceil(fine_tuning_factor*val_size))],liver_indexes['val_names_seg'][:np.int16(np.ceil(fine_tuning_factor*val_size))] 

        
        datadict_train = ImageLoader3D(liver_indexes['train_names_flair'],liver_indexes['train_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty',transform = composed_transform,data='liver')
        datadict_val = ImageLoader3D(liver_indexes['val_names_flair'],liver_indexes['val_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty', transform = ToTensor3D(True),data='liver')

    elif(which_data=='busi'):
        busi_indexes = np.load('./Data_splits/Train_val_test_42_6_x/busi_indexes.npy', allow_pickle=True).item()
        busi_indexes = helper_path_configuration(busi_indexes,system_data_path)

        datadict_train = ImageLoader2D(busi_indexes['train_names_flair'],busi_indexes['train_names_seg'],image_size=512,type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(busi_indexes['val_names_flair'],busi_indexes['val_names_seg'],image_size=512,type_of_imgs='png', transform = ToTensor2D(True),data='busi')

    elif(which_data=='idrid'):
        idrid_indexes = np.load('./Data_splits/Train_val_test_42_6_x/idrid_indexes.npy', allow_pickle=True).item()
        idrid_indexes = helper_path_configuration(idrid_indexes,system_data_path)

        datadict_train = ImageLoader2D(idrid_indexes['train_names_flair'],idrid_indexes['train_names_seg'],image_size=512,type_of_imgs='png',transform = composed_transform_2d,data='busi')
        datadict_val = ImageLoader2D(idrid_indexes['val_names_flair'],idrid_indexes['val_names_seg'],image_size=512,type_of_imgs='png', transform = ToTensor2D(True),data='busi')



    return datadict_train,datadict_val

def helper_ss_data_adaptation(which_data='brats',factor=1.0,adapt_path=None,adapt_save_path=None,model=None,hyper_parameters=None,device=0,size=(128,128,128),no_crop=False):
    sim_path = './simulation_data/Full_sim_22_11_23/'
    fine_tuning_factor = factor

    if(which_data=='wmh'):
        # White Matter Hyperintensities
        train_size = 42
        val_size = 6
        test_size = 12

        wmh_indexes = np.load('./Data_splits/Train_val_test_42_6_x/wmh_indexes.npy', allow_pickle=True).item()

        wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'] = wmh_indexes['train_names_flair'][:np.int16(np.ceil(fine_tuning_factor*train_size))],wmh_indexes['train_names_seg'][:np.int16(np.ceil(fine_tuning_factor*train_size))] 
        wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'] = wmh_indexes['val_names_flair'][:np.int16(np.ceil(fine_tuning_factor*val_size))],wmh_indexes['val_names_seg'][:np.int16(np.ceil(fine_tuning_factor*val_size))] 


        datadict_train = ImageLoader3D(wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty',transform = composed_transform,return_orig=True)
        datadict_val = ImageLoader3D(wmh_indexes['val_names_flair'],wmh_indexes['val_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty', transform = ToTensor3D(True),return_orig=True)


    elif(which_data=='brats'):
        # Brain Tumor Segmentation Challenge
        train_size = 876
        val_size = 125
        test_size = 250

        brats_indexes = np.load('./Data_splits/Train_val_test_42_6_x/brats_indexes.npy', allow_pickle=True).item()

        brats_indexes['train_names_flair'],brats_indexes['train_names_seg'] = brats_indexes['train_names_flair'][:np.int16(np.ceil(fine_tuning_factor*train_size))],brats_indexes['train_names_seg'][:np.int16(np.ceil(fine_tuning_factor*train_size))] 
        brats_indexes['val_names_flair'],brats_indexes['val_names_seg'] = brats_indexes['val_names_flair'][:np.int16(np.ceil(fine_tuning_factor*val_size))],brats_indexes['val_names_seg'][:np.int16(np.ceil(fine_tuning_factor*val_size))] 

        
        datadict_train = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty',transform = composed_transform,return_orig=True)
        datadict_val = ImageLoader3D(brats_indexes['val_names_flair'],brats_indexes['val_names_seg'],image_size=size,no_crop=no_crop,type_of_imgs='nifty', transform = ToTensor3D(True),return_orig=True)
    
    
    adapt_model = helper_model(model,which_data=which_data,hyper_parameters=hyper_parameters,device=device)
    adapt_model.load_state_dict(torch.load(adapt_path)['model_state_dict'])
    adapt_model.eval()

    trainloader = DataLoader(datadict_train, batch_size=1, shuffle=False,num_workers=1)
    valloader = DataLoader(datadict_val, batch_size=1, shuffle=False,num_workers=1)

    os.makedirs(adapt_save_path+'TrainSet/',exist_ok=True)
    os.makedirs(adapt_save_path+'ValSet/',exist_ok=True)
    print('--------------------------------------Adapting Train----------------------------------------')
    with tqdm(range(len(trainloader))) as pbar:
        for i, data in zip(pbar, trainloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                image = data['input'].to(device)
                output = adapt_model.forward(image) > 0.5
                
                image1 = nib.Nifti1Image(image[0,0].cpu().numpy(), affine=data['affine'][0])
                nib.save(image1, adapt_save_path+'TrainSet/Training'+str(i)+'_FLAIR.nii.gz')

                output1 = nib.Nifti1Image(output[0,0].cpu().numpy().astype(np.single), affine=data['affine'][0])
                nib.save(output1, adapt_save_path+'TrainSet/Training'+str(i)+'_manualmask.nii.gz')
    print('--------------------------------------Adapting Val----------------------------------------')
    with tqdm(range(len(valloader))) as pbar:
        for i, data in zip(pbar, valloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                image = data['input'].to(device)
                output = adapt_model.forward(image) > 0.5
                
                image1 = nib.Nifti1Image(image[0,0].cpu().numpy(), affine=data['affine'][0])
                nib.save(image1, adapt_save_path+'ValSet/Validation'+str(i)+'_FLAIR.nii.gz')

                output1 = nib.Nifti1Image(output[0,0].cpu().numpy().astype(np.single), affine=data['affine'][0])
                nib.save(output1, adapt_save_path+'ValSet/Valdiation'+str(i)+'_manualmask.nii.gz')

    train_names_flair = sorted(glob.glob((adapt_save_path+'TrainSet/*FLAIR.nii.gz')))  
    train_names_seg = sorted(glob.glob((adapt_save_path+'TrainSet/*manualmask.nii.gz')))

    val_names_flair = sorted(glob.glob((adapt_save_path+'ValSet/*FLAIR.nii.gz'))) 
    val_names_seg = sorted(glob.glob((adapt_save_path+'ValSet/*manualmask.nii.gz'))) 

    datadict_train = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty',transform = composed_transform,data=which_data)
    datadict_val = ImageLoader3D(val_names_flair,val_names_seg,type_of_imgs='nifty', transform = ToTensor3D(True),data=which_data)

    return datadict_train,datadict_val


def helper_model(model_type,which_data,hyper_parameters,device=0,size=(128,128,128)):
    if(model_type == 'ducknet'):
        model = DuckNet(input_channels = 1,out_classes = 1,starting_filters = 17).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = DuckNet2D(out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'nestedunet'):
        model = NestedUNet(out_channels=1,**hyper_parameters).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = NestedUNet2D(out_channels=1,**hyper_parameters).to(device)
            
    elif(model_type == 'halfunet'):
        model = HalfUNet(out_channels=1).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = HalfUNet2D(out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'resunet'):
        model = ResUNet(out_channels=1).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = ResUNet2D(out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'unet'):
        model = UNet(out_channels=1,**hyper_parameters).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = UNet2D(out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'unetr'):
        config = {
            "patch_size": 16,  # Input image size: 128x128 -> 8x8 patches
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 4 * 768, # 4 * hidden_size
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0,
            "initializer_range": 0.02,
            "image_size": (128,128,128),
            "num_classes": 1, # num_classes of binary
            "num_channels": 1,
            "qkv_bias": True,
            "use_faster_attention": True,
        }
        config['image_size'] = size
        if(which_data=='busi' or which_data=='idrid'):
            model = VITForSegmentation2D(config=config).to(device)
        else:
            model = VITForSegmentation(config=config).to(device)
            
    elif(model_type == 'slimunetr'):
        if(which_data=='busi' or which_data=='idrid'):
            model = SlimUNETR2D(in_channels=1, out_channels=1,**hyper_parameters).to(device)
        else:
            model = SlimUNETR(in_channels=1, out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'saunet'):
        model = SA_UNet(out_channels=2).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = SA_UNet2D(out_channels=1,**hyper_parameters).to(device)

    elif(model_type == 'sacunet'):
        model = SAC_UNet(out_channels=2).to(device)
        if(which_data=='busi' or which_data=='idrid'):
            model = SAC_UNet2D(out_channels=1,**hyper_parameters).to(device)

    return model

def helper_criterion(criterion_type='dice',device=0):
    if(criterion_type == 'wbce'):
        criterion= BCE_Loss_Weighted(weight=5).to(device)
    
    elif(criterion_type == 'bce'):
        criterion = nn.BCELoss().to(device)
    
    elif(criterion_type == 'dice'):
        criterion = DiceLoss().to(device)

    elif(criterion_type == 'logcoshdice'):
        criterion = LogCoshDiceLoss().to(device)

    elif(criterion_type == 'adaptive'):
        criterion = Adaptive_Region_Specific_TverskyLoss().to(device)
    
    elif(criterion_type == 'wbce + dice'):
        criterion = WBCE_DICELoss().to(device)
    
    elif(criterion_type == 'wbce + focaldice'):
        criterion = WBCE_FOCALDICELoss().to(device)

    elif(criterion_type == 'focal'):
        criterion = FocalLoss().to(device)

    elif(criterion_type == 'focal + dice'):
        criterion = FOCAL_DICELoss().to(device)

    elif(criterion_type == 'wbce + focal'):
        criterion = FOCAL_DICELoss().to(device)
    return criterion

def izipmerge(a, b):
  for i, j in zip(a,b):
    yield i
    yield j


def helper_train(epoch,model,criterion,optimizer,scheduler,train_losses,train_dices,val_losses,val_dices,trainloader,valloader,adapt_model=None,device=0):
    epoch_loss = 0
    epoch_dice = 0
    model.train()
    if(adapt_model!=None):
        adapt_model.eval()
    train_size = len(trainloader)
    val_size = len(valloader)

    if(type(trainloader) == tuple):
        train_size = len(trainloader[0])*2
        val_size = len(valloader[0])*2
        trainloader = izipmerge(*trainloader)
        valloader = izipmerge(*valloader)

    with tqdm(range(train_size)) as pbar:
        for i, data in zip(pbar, trainloader):
            torch.cuda.empty_cache()
            err = 0
            image = data['input'].to(device)
            output = model.forward(image)
            if(adapt_model!=None):
                with torch.no_grad():
                    label = adapt_model.forward(image)
            else:
                label = data['gt'].to(device)

            err = criterion(output,label,wt = label)
            dice = Dice_Score(output.cpu().detach().numpy(),label.cpu().detach().numpy())
            
            model.zero_grad()
            err.backward()
            optimizer.step()
            pbar.set_postfix(Train_Loss = np.round(err.cpu().detach().numpy().item(), 5),Train_Dice = np.round(dice, 5))
            pbar.update(0)
            epoch_loss += err.item()
            epoch_dice += dice
            del label
            del err

        train_losses.append([epoch_loss/train_size])
        train_dices.append([epoch_dice/train_size])
        print('Training Loss and Dice at epoch {} is : Total {} and {}'.format(epoch,*train_losses[-1],*train_dices[-1]))

    epoch_loss = 0
    epoch_dice = 0
    model.eval()
    with tqdm(range(val_size)) as pbar:
        for i, data in zip(pbar, valloader):
            torch.cuda.empty_cache()
            err = 0
            with torch.no_grad():
                image = data['input'].to(device)
                output = model.forward(image)
            
                if(adapt_model!=None):
                    with torch.no_grad():
                        label = adapt_model.forward(image)
                else:
                    label = data['gt'].to(device)

                err = criterion(output,label,wt =label)
                dice = Dice_Score(output.cpu().detach().numpy(),label.cpu().detach().numpy())
                del image
                del label

            pbar.set_postfix(Val_Loss = np.round(err.cpu().detach().numpy().item(), 5),Val_Dice=np.round(dice, 5))
            pbar.update(0)
            epoch_loss += err.item()
            epoch_dice += dice
            del err

        val_losses.append([epoch_loss/val_size])
        val_dices.append([epoch_dice/val_size])
        print('Validation Loss and Dice at epoch {} is : Total {} and {}'.format(epoch,*val_losses[-1],*val_dices[-1]))
    
    scheduler.step(*val_dices[-1])
    return epoch_loss,epoch_dice

def helper_save_model(epoch,model,optimizer,epoch_loss,epoch_dice,scheduler,model_type,best_dice=False):
    if(not best_dice):
        loss_type = '_state_dict'
    else:
        loss_type = '_state_dict_best_dice'
    torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss,
    'dice':epoch_dice,
    'lr_scheduler_state_dict':scheduler.state_dict(),
    }, model_type+loss_type+str(epoch)+'.pth')