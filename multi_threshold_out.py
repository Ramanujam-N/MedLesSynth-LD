import nibabel as nib
import numpy as np
from ModelArchitecture.metrics import *
import glob
from tqdm import tqdm
import pandas as pd
path =  'images/21_06_24/Data_Augmentation/lits/texshapes_onsamesize'

pred_paths = sorted(glob.glob(path+'/*wothresh*'))
gt_paths = sorted(glob.glob(path+'/*manualmask*'))

dice_list = []
hau_list = []
tpr_list = []
tnr_list = []

thresholds = np.arange(0.1,1,0.1)
j =5
with tqdm(range(len(pred_paths))) as pbar:
    for i in pbar:
        pred = nib.load(pred_paths[i]).get_fdata()
        label = nib.load(gt_paths[i]).get_fdata()
        dice = Dice_Score(pred,label,threshold=thresholds[j])
        hau_acc = Hausdorff_score(pred,label,threshold=thresholds[j])
        tpr_acc = TPR(pred,label,threshold=thresholds[j])
        tnr_acc = TNR(pred,label,threshold=thresholds[j])

        dice_list.append(dice)
        hau_list.append(hau_acc)
        tpr_list.append(tpr_acc)
        tnr_list.append(tnr_acc)
        
        pbar.update(0)
        pd.DataFrame({'Dice':dice_list,'Hausdorff':hau_list,'TPR':tpr_list,'TNR':tnr_list,}).to_csv(f'./csv_files/21_06_24/Data_Augmentation/lits/texshapes_onsamesize/thresh_{thresholds[j]}_186.csv')

# dice_list = np.array(dice_list)
# hau_list = np.array(hau_list)
# tpr_list = np.array(tpr_list)
# tnr_list = np.array(tnr_list)

# for i in range(10):
#     print(f'For thresh {thresholds[i]} DICE {dice_list[:,i].mean()}({dice_list[:,i].std()}) TPR {tpr_list[:,i].mean()}({tpr_list[:,i].std()}) \
#           HD95 {tpr_list[:,i].mean()}({tpr_list[:,i].std()}) TNR {tnr_list[:,i].mean()}({tnr_list[:,i].std()})')
