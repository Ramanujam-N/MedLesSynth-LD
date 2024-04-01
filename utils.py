import numpy as np
from torch.utils.data import ConcatDataset
import nibabel as nib
import json
import matplotlib.pyplot as plt
import glob
from scipy.ndimage import gaussian_filter
from ImageLoader.Ablation import LesionGeneration
from ImageLoader.ImageLoader3D import ImageLoader3D
import matplotlib.animation as animation
from ModelArchitecture.Transformations import ToTensor3D
np.random.seed(0)
import numpy as np
from ImageLoader.ImageLoader3D import ImageLoader3D
from torch.utils.data import DataLoader,ConcatDataset
from ModelArchitecture.metrics import *
from ModelArchitecture.Transformations import *
import torch
from tqdm import tqdm
from helper_datadict import helper_resize
import sys
size = (128,128,128)
import nibabel as nib
from PIL import Image
import os

np.random.seed(0)

def wmh_training_index():

        # 60 - 42 6 12
        wmh_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/WMH_nifti_versions/*FLAIR*')))
        gt_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/WMH_nifti_versions/*manualmask*')))

        permutation = np.random.choice(len(wmh_indexes), len(wmh_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        wmh_indexes = wmh_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        wmh_dictionary = {'train_names_flair':wmh_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':wmh_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':wmh_indexes[48:],'test_names_seg':gt_indexes[48:]}
        print(len(wmh_dictionary['train_names_flair']),len(wmh_dictionary['val_names_flair']),len(wmh_dictionary['test_names_flair']))
        np.save('./wmh_training_indexes.npy',wmh_dictionary)


def brats_2021_43_training_index():

        # 1251 - 42 6 1203
        brats_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*flair*7*')))
        gt_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*seg*7*')))

        permutation = np.random.choice(len(brats_indexes), len(brats_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        brats_indexes = brats_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        brats_dictionary = {'train_names_flair':brats_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':brats_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':brats_indexes[48:],'test_names_seg':gt_indexes[48:]}
        print(len(brats_dictionary['train_names_flair']),len(brats_dictionary['val_names_flair']),len(brats_dictionary['test_names_flair']))
        np.save('./brats_2021_42_training_indexes.npy',brats_dictionary)


def liver_index():
        liver_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/Liver/Task03_Liver/imagesTr/*')))
        gt_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/Liver/Task03_Liver/labelsTr/*')))

        tumour_indexes = []
        for i in range(len(gt_indexes)):
                #print(i,np.sum(nib.load(gt_indexes[i]).get_fdata()==2)>0)
                if(np.sum(nib.load(gt_indexes[i]).get_fdata()==2)>0):
                        tumour_indexes.append(i)

        tumour_indexes = np.array(tumour_indexes)

        liver_indexes = liver_indexes[tumour_indexes]
        gt_indexes = gt_indexes[tumour_indexes]

        permutation = np.random.choice(len(liver_indexes), len(liver_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        liver_indexes = liver_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        liver_dictionary = {'train_names_flair':liver_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':liver_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':liver_indexes[48:],'test_names_seg':gt_indexes[48:]}
        
        np.save('./liver_indexes.npy',liver_dictionary)


def busi_index():
        bus_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/BUSI/Dataset_BUSI/Dataset_BUSI_with_GT/benign/*).png')))
        gt_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/BUSI/Dataset_BUSI/Dataset_BUSI_with_GT/benign/*mask.png')))

        permutation = np.random.choice(len(bus_indexes), len(bus_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        bus_indexes = bus_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        busi_dictionary = {'train_names_flair':bus_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':bus_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':bus_indexes[48:],'test_names_seg':gt_indexes[48:]}
        
        np.save('./busi_indexes.npy',busi_dictionary)


def idrid_index():

        idrid_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/IDRiD/Dataset/Images/*')))
        gt_indexes = np.array(sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/IDRiD/Dataset/Masks/*')))

        permutation = np.random.choice(len(idrid_indexes), len(idrid_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        bus_indexes = idrid_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        idrid_dictionary = {'train_names_flair':idrid_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':idrid_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':idrid_indexes[48:],'test_names_seg':gt_indexes[48:]}
        np.save('./idrid_indexes.npy',idrid_dictionary)



def clean_liver():
        liver_indexes = sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/CleanLiver/*.nii.gz'))
        mask_indexes = sorted(glob.glob('/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/LabData/CleanLiver/*masks*.nii.gz'))
        temp = []
        for i in liver_indexes:
                if(i not in mask_indexes):
                        temp.append(i)
        liver_indexes = temp

        cleanliver_dictionary = {'images':liver_indexes,'masks':mask_indexes}
        print(cleanliver_dictionary)
        np.save('./clean_liver_indexes.npy',cleanliver_dictionary)


        