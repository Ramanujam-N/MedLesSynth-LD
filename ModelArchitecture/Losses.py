import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage
import numpy as np
import einops
import einops.layers.torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# def loss_weight_map(gnd_map):
#     wt_map = torch.zeros_like(gnd_map)
#     for k in range(wt_map.shape[0]):
#         labelled_map = skimage.measure.label(gnd_map[k,1],background = 0)
#         regions = skimage.measure.regionprops(labelled_map)
        
#         for i in regions:
#             centroid = i.centroid
#             coords = i.coords
#             wts = np.sqrt(np.sum((coords - centroid)**2,axis=-1))
#             wts -= wts.min()
#             wts /= wts.max() + 1e-7
#             for mm in range(wts.shape[0]):
#                 wt_map[k,0,*coords[mm]] = 1 - wts[mm]
#                 wt_map[k,1,*coords[mm]] = 1 - wts[mm]

#     return wt_map

class GRAM_Loss(nn.Module):
    def __init__(self,vgg_enc):
        self.vgg_enc = vgg_enc
    def gram(self,x):
        b,c,h,w,d = x.size()
        x = x.view(b*c, -1)
        return torch.mm(x, x.t())  
    def forward(self,x,y):
        x_feats = self.vgg_enc(x)
        y_feats = self.vgg_enc(y)
        loss = 0
        for i in range(len(x_feats)):
            loss+=nn.functional.mse_loss(self.gram(x_feats[i]),self.gram(y_feats[i]))
        return loss

class Frequency_loss(nn.Module):
    def __init__(self,weight):
        super().__init__()
        self.weights = weight
    def forward(self,x,y):
        return torch.sum(self.weights*torch.abs(x-y))


class MS_SSIMLoss(nn.Module):
    def __init__(self,data_range = 1.0,win_size=5,win_sigma=1.5,channel=1,spatial_dims=3):
        super(MS_SSIMLoss,self).__init__()
        self.eps = 1e-7
        self.data_range = data_range
        self.win_size = win_size
        self.win_sigma = win_sigma
        self.channel = channel
        self.spatial_dims = spatial_dims
        self.ms_ssim = MS_SSIM(data_range = self.data_range,win_size = self.win_size,win_sigma = self.win_sigma,channel = self.channel,spatial_dims =self.spatial_dims)
    def forward(self,x,y):
        return 1-self.ms_ssim(x,y)

class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss, self).__init__()
        self.eps = 1e-7
    def forward(self,x,target):
        x = torch.sigmoid(x)
        return torch.mean((-target*torch.log(x+self.eps) - (1-target)*torch.log(1-x+self.eps)))
    
class BCE_Loss_Weighted(nn.Module):
    def __init__(self,weight=None):
        super(BCE_Loss_Weighted, self).__init__()
        self.eps = 1e-7
        self.weight = torch.tensor(weight)
    def forward(self,x,target,wt):
        #self.weight = torch.tensor(wt)
        wt = torch.nan_to_num(wt,0,0,0)*self.weight
        if(wt.sum()==0):
            wt =1 
        
        return torch.mean((-wt*target*torch.log(x+self.eps) - (1-target)*torch.log(1-x+self.eps)))


class DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.eps = 1e-7

    def forward(self, x, target,wt=1):
        #num_classes = target.shape[1]  # Channels first
        #target = target.type(x.type())
        smooth = 1
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x.double() * target, dims)
        cardinality = torch.sum(x.double() + target, dims)

        dice_score = ((2. * intersection + smooth) / (cardinality + smooth))
        return (1 - dice_score.mean())


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1,):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    

       
        s_Dice = (TP + smooth) / ((inputs**2).sum() + (targets**2).sum() + smooth)  
        
        return 1 - s_Dice

class LogCoshDiceLoss(nn.Module):
    def __init__(self,weight=None):
        super().__init__()

    def forward(self, x, target, wt = 1):
        smooth = 1
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_score = ((2. * intersection + smooth) / (cardinality + smooth))
        dice_loss = (1 - dice_score)
        
        return torch.log((torch.exp(dice_loss) + torch.exp(-dice_loss)) / 2.0).mean()


def gram_matrix_new(y):
    b, ch, h, w, d = y.shape
    return torch.einsum('bihwd,bjhwd->bij', [y, y]) / (h * w *d)

class StyleLoss(nn.Module):
    def __init_(self,):
        pass
    def forward(self, x, y):
        gram_x = gram_matrix_new(x)
        gram_y = gram_matrix_new(y)
        return torch.square(gram_x-gram_y).mean()


class ContrastiveLoss(nn.Module):
    def __init__(self,temperature=0.05,batch_size=32,n_views=2,device='cuda:0'):
        super(ContrastiveLoss,self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.n_views = n_views
        self.device = device
    def forward(self,features):
        self.batch_size = features.shape[0]//2
        labels = torch.cat([torch.arange(self.batch_size) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels
    
class PatchwiseSSIMLoss(nn.Module):
    
    def __init__(self, k1=0.01, k2=0.03, data_range = 1.0, patch_size=16, alpha=1, beta=1, gamma=1):
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.data_range = data_range
        self.c1 = (self.k1*self.data_range)**2
        self.c2 = (self.k2*self.data_range)**2
        self.c3 = self.c2/2
        self.patch_size=patch_size
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma
        self.reduce_label = einops.layers.torch.Reduce('b (h h2) (w w2) (d d2) -> (b h w d)','mean',h2=self.patch_size,w2=self.patch_size,d2=self.patch_size) 
        self.reduce_mean = einops.layers.torch.Reduce('b c (h h2) (w w2) (d d2) -> (b c h w d)','mean',h2=self.patch_size,w2=self.patch_size,d2=self.patch_size)
        self.img2patch = einops.layers.torch.Rearrange('b c (h h2) (w w2) (d d2) -> (b h w d) (c h2 w2 d2)',h2=self.patch_size,w2=self.patch_size,d2=self.patch_size)
    
    def forward(self,x,x_label,y):

        # we get ssim score between x and y let's assume this is for the real
        mean_x = self.reduce_mean(x)
        mean_y = self.reduce_mean(y)
        mean_xy = self.reduce_mean(x*y)

        var_x = self.reduce_mean(x**2) - mean_x**2
        var_y = self.reduce_mean(y**2) - mean_y**2
        std_x = torch.sqrt(var_x) 
        std_y = torch.sqrt(var_y)
        covar = mean_xy - mean_x*mean_y

        l = (2*mean_x*mean_y + self.c1)/(mean_x**2 + mean_y**2 +self.c1)
        c = (2*std_x*std_y +self.c2)/(var_x + var_y +self.c2)
        s = (covar +self.c3)/(std_x*std_y+ self.c3)

        ssim_xy = (l**self.alpha * c**self.beta * s*self.gamma).mean(0)
        ssim_xy = torch.nn.functional.relu(ssim_xy)

        # we get ssim score between x_anomaly patch with every patch in x_normal and this is maximized
        # Need to figure out covariance if we want to construct structure map we can but it would be pretty slow
        # Let us see how we can go about this now.
        
        positive_patch_mask = self.reduce_label(x_label[:,1]) > 0.1
        negative_patch_mask = self.reduce_label(x_label[:,1]) <= 0.1

        mean_x_pos = mean_x[positive_patch_mask]
        mean_x_neg = mean_x[negative_patch_mask]
        mean_x_posneg = mean_x_pos.unsqueeze(1)@mean_x_neg.unsqueeze(0)

        var_x_pos = var_x[positive_patch_mask]
        var_x_neg = var_x[negative_patch_mask]
        std_x_pos = torch.sqrt(var_x[positive_patch_mask])
        std_x_neg = torch.sqrt(var_x[negative_patch_mask])
        std_x_posneg = std_x_pos.unsqueeze(1)@std_x_neg.unsqueeze(0)


        l_n_abn = (2*mean_x_posneg+self.c1)/(torch.stack([mean_x_pos**2,]*len(mean_x_neg),axis=1) + torch.stack([mean_x_neg**2,]*len(mean_x_pos),axis=1).T+self.c1 )
        c_n_abn = (2*std_x_posneg+self.c2)/(torch.stack([var_x_pos,]*len(var_x_neg),axis=1) + torch.stack([var_x_neg**2,]*len(var_x_pos),axis=1).T+self.c2 )
        # s_n_abs = (covar + self.c3)/(std_x_posneg + self.c3)

        ssim_nabn = (l_n_abn**self.alpha * c_n_abn**self.beta * 1*self.gamma).mean(0)
        

        #print(ssim_xy.shape,ssim_xy.shape,ssim_xy.mean(),ssim_nabn.mean())
        return  (-ssim_nabn.mean()) 

        

class PatchwiseSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, device ='cuda:0'):
        super().__init__()
        self.temperature = temperature
        self.struct_element = np.ones((5, 5, 5), dtype=bool)
        self.device = device
        self.coefficient = 1
        self.image_size=128
        self.patch_size=16
        self.batch_size=4
        self.channel_size=1
        self.reduce = einops.layers.torch.Reduce('b (h h2) (w w2) (d d2) -> (b h w d)','mean',h2=self.patch_size,w2=self.patch_size,d2=self.patch_size)
        self.img2patch = einops.layers.torch.Rearrange('b c (h h2) (w w2) (d d2) -> (b h w d) (c h2 w2 d2)',h2=self.patch_size,w2=self.patch_size,d2=self.patch_size)
        self.patch2image = einops.layers.torch.Rearrange('(b c h w d) (h2 w2 d2) -> b c (h h2) (w w2) (d d2)',b=self.batch_size,c=self.channel_size,h=self.image_size//self.patch_size,w=self.image_size//self.patch_size,d=self.image_size//self.patch_size,h2=self.patch_size,w2=self.patch_size,d2=self.patch_size)

    def forward(self, Zs, pixel_mask, subtracted_mask=None, brain_mask=None):
    
        positive_patch_mask = self.reduce(pixel_mask[:,1]) > 0.1
        negative_patch_mask = self.reduce(pixel_mask[:,0]) > 0.9

        patched_image = self.img2patch(Zs)
        positive_patches = patched_image[positive_patch_mask,:]
        negative_patches = patched_image[negative_patch_mask,:]
        

        patches = torch.cat([positive_patches, negative_patches])
        labels = torch.tensor([1] * positive_patches.shape[0] + [0] * negative_patches.shape[0]).to(self.device)

        patches = F.normalize(patches)
        dot = torch.matmul(patches, patches.T)
        dot = torch.div(dot, self.temperature)
        #dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(labels, labels.unsqueeze(1))
        class_mask[torch.arange(len(labels)), torch.arange(len(labels))] = False

        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        # positive_mask = exp * class_mask + (~class_mask).float()
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -torch.mean(full_term) * self.coefficient
        return loss

class VoxelwiseSupConLoss_inImage(nn.Module):
    def __init__(self, temperature=0.07, device='cpu', num_voxels=10500):
        super().__init__()
        self.temperature = temperature
        self.struct_element = np.ones((5, 5, 5), dtype=bool)
        self.device = device
        self.max_pixels = num_voxels
        self.coefficient = 1
    
    def forward(self, Zs, pixel_mask, subtracted_mask=None, brain_mask=None):

        number_of_features = Zs.shape[1]
        positive_mask = (pixel_mask[:, 1] == 1).squeeze(0)

        if subtracted_mask is not None:
            negative_mask = (subtracted_mask == 1).squeeze(0, 1)
        elif brain_mask is not None:
            negative_mask = torch.logical_and(brain_mask[:, 0] == 1, pixel_mask[:, 0] == 1).squeeze(0)
        else:
            negative_mask = (pixel_mask[:, 0] == 1).squeeze(0)

        positive_pixels = Zs[:, :, positive_mask].permute(0, 2, 1).reshape(-1, number_of_features)
        negative_pixels = Zs[:, :,negative_mask].permute(0, 2, 1).reshape(-1, number_of_features)

        if positive_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randint(0, positive_pixels.size(0), size=(self.max_pixels,))
            positive_pixels = positive_pixels[random_indices]
        
        if positive_pixels.shape[0] < negative_pixels.shape[0]:
            random_indices = torch.randint(0, negative_pixels.size(0), size=(positive_pixels.shape[0],))
            negative_pixels = negative_pixels[random_indices]
        elif negative_pixels.shape[0] > self.max_pixels:
            random_indices = torch.randint(0, negative_pixels.size(0), size=(self.max_pixels,))
            negative_pixels = negative_pixels[random_indices]

        pixels = torch.cat([positive_pixels, negative_pixels])
        labels = torch.tensor([1] * positive_pixels.shape[0] + [0] * negative_pixels.shape[0]).to(self.device)

        pixels = F.normalize(pixels)
        dot = torch.matmul(pixels, pixels.T)
        dot = torch.div(dot, self.temperature)
        dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(labels, labels.unsqueeze(1))
        class_mask[torch.arange(len(labels)), torch.arange(len(labels))] = False

        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        # positive_mask = exp * class_mask + (~class_mask).float()
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / len(labels)) * torch.sum(full_term) * self.coefficient
        
        return loss

class SupervisedContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, Zs, class_labels):

        number_of_samples = Zs.shape[0]
        # class_labels = 1 - torch.argmax(class_labels, dim=1)
        Zs = F.normalize(Zs)
        dot = torch.matmul(Zs, Zs.T)
        dot = torch.div(dot, self.temperature)
        # dot = F.normalize(dot)
        exp = torch.exp(dot)

        class_mask = torch.eq(class_labels, class_labels.unsqueeze(1))
        class_mask[torch.arange(number_of_samples), torch.arange(number_of_samples)] = False
        
        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        negative_mask = exp * (~class_mask)

        denominator = torch.sum(negative_mask, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / (number_of_samples)) * torch.sum(full_term)

        # loss = 0
        # for row_idx in range(number_of_samples):
        #     row = exp[row_idx]
        #     # print(f'Positive row sum: {torch.sum(row[class_mask[row_idx]])}')
        #     # print(f'Negative row sum: {torch.sum(row[~class_mask[row_idx]])}')
        #     # print(f'Row diagonal: {row[row_idx]}')
        #     denominator = torch.sum(row[~class_mask[row_idx]]) - row[row_idx]
        #     temp = torch.log(row[class_mask[row_idx]]) - torch.log(denominator)
        #     temp = torch.sum(temp)
        #     temp = (-1 / (number_of_samples-1)) * temp
        #     loss += temp
        
        return loss






class DiceLossWeighted(nn.Module):
    def __init__(self, weight=None):
        super(DiceLossWeighted, self).__init__()
        self.eps = 1e-7
        self.weight = weight

    def forward(self, x, target,wt = 1):
        #num_classes = target.shape[1]  # Channels first
        #target = target.type(x.type())
        self.weight = wt
        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(x * target, dims)
        cardinality = torch.sum(x + target, dims)

        dice_loss = ((2. * intersection + self.eps) / (cardinality + self.eps))
        return (1 - (dice_loss*self.weight).mean())

def determine_dice_metric(pred, target):
    smooth = 1.

    n_classes = pred.shape[1]  # b*c*h*w
    avg_dice = 0.0
    for i in range(n_classes):
        pred_vect = pred[:, i, :, :].contiguous().view(-1)
        target_vect = target[:, i, :, :].contiguous().view(-1)
        intersection = (pred_vect * target_vect).sum()
        dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
        avg_dice += dice
    return avg_dice / n_classes

from  ModelArchitecture.Losses_unified_focal import AsymmetricFocalTverskyLoss

class DICELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = AsymmetricFocalTverskyLoss(delta=0.5,gamma=0)(pred,target)
        return dice

class FocalDICELoss(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = AsymmetricFocalTverskyLoss(delta=0.5,gamma=2)(pred,target)
        return dice

class WBCE_DICELoss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        #print(wt.shape)
        dice = DiceLoss()(pred[:,1],target[:,1])
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce

class WBCE_FOCALDICELoss(nn.Module):
    def __init__(self,):
        super().__init__()

        pass
    def forward(self,pred,target,wt=1):
        dice = AsymmetricFocalTverskyLoss(delta=0.5)(pred,target) # this becomes dice
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return dice + wbce


class FOCAL_DICELoss(nn.Module):
    def __init__(self,focal_param={},tversky_param={}):
        super().__init__()
        self.focal_param =focal_param
        self.tversky_param = tversky_param
        pass
    def forward(self,pred,target,wt=1):
        dice = TverskyLoss(**self.tversky_param)(pred,target) # this becomes dice
        focal = FocalLoss(**self.focal_param)(pred,target)
        return dice + focal
    
class WBCE_FOCALLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        pass
    def forward(self,pred,target,wt=1):
        focal = FocalLoss()(pred,target) # this becomes dice
        wbce = BCE_Loss_Weighted(weight=5)(pred,target,wt)
        return focal+wbce
    
#################################################################################################
# Losses Taken from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
#################################################################################################
ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, alpha=ALPHA, gamma=GAMMA, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets, smooth=1,wt=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss
    
#PyTorch
ALPHA = 0.5
BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, alpha=ALPHA, beta=BETA, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets, smooth=1,):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + self.alpha*FP + self.beta*FN + smooth)  
        
        return 1 - Tversky
    
ALPHA = 0.5
BETA = 0.5
GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
CE_RATIO = 0.5 #weighted contribution of modified CE loss compared to Dice loss

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, eps=1e-9):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, eps, 1.0 - eps)       
        out = - (ALPHA * ((targets * torch.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
        
        return combo