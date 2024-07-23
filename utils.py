import numpy as np
from torch.utils.data import ConcatDataset
import nibabel as nib
import matplotlib.pyplot as plt
import glob
np.random.seed(0)
import numpy as np
from ModelArchitecture.metrics import *
from ModelArchitecture.Transformations import *
import nibabel as nib

np.random.seed(0)

def wmh_training_index():

        # 60 - 42 6 12
        wmh_indexes = np.array(sorted(glob.glob('./WMH_nifti_versions/*FLAIR*')))
        gt_indexes = np.array(sorted(glob.glob('./WMH_nifti_versions/*manualmask*')))

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
        brats_indexes = np.array(sorted(glob.glob('./RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*flair*7*')))
        gt_indexes = np.array(sorted(glob.glob('./RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021/**/*seg*7*')))

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
        liver_indexes = np.array(sorted(glob.glob('./Liver/Task03_Liver/imagesTr/*')))
        gt_indexes = np.array(sorted(glob.glob('./Liver/Task03_Liver/labelsTr/*')))

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
        bus_indexes = np.array(sorted(glob.glob('./BUSI/Dataset_BUSI/Dataset_BUSI_with_GT/benign/*).png')))
        gt_indexes = np.array(sorted(glob.glob('./BUSI/Dataset_BUSI/Dataset_BUSI_with_GT/benign/*mask.png')))

        permutation = np.random.choice(len(bus_indexes), len(bus_indexes), replace=False).astype(np.int16)   
        print(permutation)    
        bus_indexes = bus_indexes[permutation]
        gt_indexes = gt_indexes[permutation]

        busi_dictionary = {'train_names_flair':bus_indexes[:42],'train_names_seg':gt_indexes[:42],
                        'val_names_flair':bus_indexes[42:48],'val_names_seg':gt_indexes[42:48],
                        'test_names_flair':bus_indexes[48:],'test_names_seg':gt_indexes[48:]}
        
        np.save('./busi_indexes.npy',busi_dictionary)

