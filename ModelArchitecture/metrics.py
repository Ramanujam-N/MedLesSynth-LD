import numpy as np
import scipy as sc
import medpy.metric
import skimage
import matplotlib.pyplot as plt
def Dice_Score(pred,target,threshold=0.5):
    smooth = 1
    pred_cmb = (pred > threshold).astype(float)
   
    true_cmb = (target > 0).astype(float)

    pred_vect = pred_cmb.reshape(-1)
    true_vect = true_cmb.reshape(-1)
    intersection = np.sum(pred_vect * true_vect)
    if np.sum(pred_vect) == 0 and np.sum(true_vect) == 0:
        dice_score = (2. * intersection + smooth) / (np.sum(pred_vect) + np.sum(true_vect) + smooth)
    else:
        dice_score = (2. * intersection) / (np.sum(pred_vect) + np.sum(true_vect))
        #print('intersection, true_vect, pred_vect')
        #print(intersection, np.sum(true_vect), np.sum(pred_vect))
    return dice_score

# Voxel Wise
def TPR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1) 
    target = (target > 0).astype(float)

    TP = np.sum(target*(pred>threshold))
    P = np.sum(target)
    return (TP+eps)/(P+eps)

# Voxel Wise
def TNR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    TN = np.sum((1-target)*(pred<=threshold))
    N = np.sum(1-target)
    return (TN+eps)/(N+eps)

# Voxel Wise
def FPR(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    FP = np.sum((1-target)*(pred>threshold))
    N = np.sum(1-target)
    return (FP+eps)/(N+eps)

def Hausdorff_score(pred,target,threshold=0.5):
    # eps = 1e-7
    pred = (pred>threshold).astype(float)
    target = (target > 0).astype(float)
    
    pred_label,pred_cc = skimage.measure.label(pred*255,background = 0,return_num=True)
    target_label,target_cc = skimage.measure.label(target*255,background = 0,return_num=True)
    pred_w_intersect = np.zeros_like(pred_label)

    for i in range(1,pred_cc+1):
        prediction = pred_label == i

        if((prediction*target).sum()):
            pred_w_intersect = np.logical_or(prediction,pred_w_intersect)    
    pred = pred_w_intersect

    if(pred.sum()==0):
        if(len(pred.shape)==2):
            pred[0,0] = 1
        else:
            pred[0,0,0] = 1
    hd95 = medpy.metric.binary.hd95(result=pred, reference = target)
    return np.array(hd95)/max(pred.shape)