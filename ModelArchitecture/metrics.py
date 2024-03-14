import numpy as np

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

# Voxel Wise
def F1_score(pred,target,threshold=0.5):
    eps = 1e-7
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    target = (target > 0).astype(float)

    TP = np.sum(target*(pred>threshold))
    FP = np.sum((1-target)*(pred>threshold))
    FN = np.sum(target*(pred<threshold))
    
    return (2*TP+eps)/(2*TP+FP+FN+eps)