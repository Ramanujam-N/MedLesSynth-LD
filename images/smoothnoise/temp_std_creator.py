import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import skimage
from PIL import Image


# for i in range(70):
#     images = glob.glob(r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/30_03_24/Fine_Tuning_Data_Augmentation/lits/*smooth*/*Testing{}_*wothresh*'.format(i))
#     # manual_masks = glob.glob(r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/12_03_24/Supervised/brats/*smooth*/*Testing{}_*manualmask*'.format(i))

#     img_list = [nib.load(img).get_fdata()  for img in images]
#     # manual_mask = nib.load(manual_masks[0]).get_fdata()
#     std_map = np.std(img_list,axis=0)

#     std_nib = nib.Nifti1Image(std_map, affine = nib.load(images[0]).affine)
#     nib.save(std_nib,r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/30_03_24/Fine_Tuning_Data_Augmentation/lits/std_maps/Testing{}_std_map.nii.gz'.format(i))
#     print(i)

for i in range(389):
    images = glob.glob(r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/12_03_24/Supervised/busi/*smooth*/*Testing{}_*wothresh*'.format(i))
    # manual_masks = glob.glob(r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/12_03_24/Supervised/brats/*smooth*/*Testing{}_*manualmask*'.format(i))

    img_list = [np.array(Image.open(img).convert('L'))  for img in images]
    # manual_mask = nib.load(manual_masks[0]).get_fdata()
    std_map = np.std(img_list,axis=0)

    std_map = np.uint8(std_map)

    # std_nib = std_map
    # nib.save(std_nib,r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/30_03_24/Fine_Tuning_Data_Augmentation/busi/std_maps/Testing{}_std_map.nii.gz'.format(i))
    Image.fromarray(std_map).save(r'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f/projects/OOD_other/Anosynth_V1/images/smoothnoise/12_03_24/Supervised/busi/std_maps/Testing{}_std_map.png'.format(i))
    print(i)