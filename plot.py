import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
from ImageLoader.ImageLoader3D import ImageLoader3D
from ModelArchitecture.Transformations import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from ModelArchitecture.Encoders import ClassificationModel
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from einops import rearrange, reduce
from skimage.measure import label,regionprops

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors

path = './results/Recon_4_11_23/HalfUNet_ssim + l1_loss.npy'
def plot_loss_curves(path):
    x = np.load(path)
    plt.title(' Recon Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(x[0,:])
    plt.plot(x[1,:])
    #plt.yticks(np.arange(min(x[1,:]), max(x[1,:])+1, 0.5))
    #plt.xticks(np.arange(0,len(x[1,:]), 1))
    #plt.grid(True)
    plt.legend(['train','val'])
    plt.savefig('./plots/' + path[10:-4] + '.png')
    plt.show()


torch.manual_seed(6)
"""
wmh_indexes = np.load('./wmh_indexes_new.npy', allow_pickle=True).item()
real_abnormal = ImageLoader3D(wmh_indexes['train_names_flair'],wmh_indexes['train_names_seg'],image_size=(128,128,128),type_of_imgs='nifty',data='WMH',no_crop=False)
real_abnormal_test = ImageLoader3D(wmh_indexes['test_names_flair'],wmh_indexes['test_names_seg'],image_size=(128,128,128),type_of_imgs='nifty',data='wmh',no_crop=False)

train_names_flair = sorted(glob.glob(('./simulation_data/26_02_24/'+'wmh'+'/TrainSet/*[0-9]_FLAIR.nii.gz')))[:42]
train_names_seg = sorted(glob.glob(('./simulation_data/26_02_24/'+'wmh'+'/TrainSet/*[0-9]_manualmask.nii.gz')))[:42]
sim_abnormal = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', transform = None)   


real_ab_list = []
real_n_list = []

real_ab_test_list = []
real_n_test_list = []

# sim_n_list = []
sim_ab_list = []
sim_n_list = []

# rearrange('(h h2) (w w2) (d d2) c-> (b h w d) (c h2 w2 d2)',h2=16,w2=16,d2=16)
# reduce(real_abnormal[i]['gt'][:,:,:,1],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=16,w2=16,d2=16) >0.1



with tqdm(range(len(real_abnormal_test))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(real_abnormal_test[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(real_abnormal_test[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            real_ab_test_list.append(ab_patch.reshape(-1))
            
            #print(ab_patch.shape,cent,pad_shape,new_cent)
        n_patch_mask = reduce(real_abnormal_test[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(real_abnormal_test[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        
        real_n_test_list.append(n_patches)
        pbar.update(0)

real_ab_test_list = np.stack(real_ab_test_list)
real_n_test_list = np.concatenate(real_n_test_list)
np.save('Patchestrain_real_ab_test_wmh_26_02_24_32.npy',real_ab_test_list)
np.save('Patchestrain_real_n_test_wmh_26_02_24_32.npy',real_n_test_list)


with tqdm(range(len(real_abnormal))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(real_abnormal[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(real_abnormal[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            real_ab_list.append(ab_patch.reshape(-1))

            # print(ab_patch.shape)
        n_patch_mask = reduce(real_abnormal[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(real_abnormal[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        
        real_n_list.append(n_patches)
        pbar.update(0)

real_ab_list = np.stack(real_ab_list)
real_n_list = np.concatenate(real_n_list)
np.save('Patchestrain_real_ab_wmh_26_02_24_32.npy',real_ab_list)
np.save('Patchestrain_real_n_wmh_26_02_24_32.npy',real_n_list)


with tqdm(range(len(sim_abnormal))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(sim_abnormal[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(sim_abnormal[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            sim_ab_list.append(ab_patch.reshape(-1))

            # print(ab_patch.shape)
        n_patch_mask = reduce(sim_abnormal[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(sim_abnormal[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        sim_n_list.append(n_patches)
        pbar.update(0)


sim_ab_list = np.stack(sim_ab_list)
sim_n_list = np.concatenate(sim_n_list)
np.save('Patchestest_sim_ab_wmh_26_02_24_32.npy',sim_ab_list)
np.save('Patchestest_sim_n_wmh_26_02_24_32.npy',sim_n_list)

# print('Sizes are ',real_ab_test_list.shape, real_n_test_list.shape ,real_ab_list.shape, real_n_list.shape,sim_ab_list.shape,sim_n_list.shape)




"""


######################################################################################





combined_brats_samples = 250
brats_indexes = np.load('./brats_2021_42_training_indexes.npy', allow_pickle=True).item()
real_abnormal = ImageLoader3D(brats_indexes['train_names_flair'],brats_indexes['train_names_seg'],image_size=(128,128,128),type_of_imgs='nifty',data='brats',no_crop=False)
real_abnormal_test = ImageLoader3D(brats_indexes['test_names_flair'][:combined_brats_samples],brats_indexes['test_names_seg'][:combined_brats_samples],image_size=(128,128,128),type_of_imgs='nifty',data='brats',no_crop=False)

train_names_flair = sorted(glob.glob(('./simulation_data/26_02_24/'+'brats'+'/TrainSet/*[0-9]_FLAIR.nii.gz')))[:42]
train_names_seg = sorted(glob.glob(('./simulation_data/26_02_24/'+'brats'+'/TrainSet/*[0-9]_manualmask.nii.gz')))[:42]
sim_abnormal = ImageLoader3D(train_names_flair,train_names_seg,type_of_imgs='nifty', transform = None)   


real_ab_list = []
real_n_list = []

real_ab_test_list = []
real_n_test_list = []

# sim_n_list = []
sim_ab_list = []
sim_n_list = []

# rearrange('(h h2) (w w2) (d d2) c-> (b h w d) (c h2 w2 d2)',h2=16,w2=16,d2=16)
# reduce(real_abnormal[i]['gt'][:,:,:,1],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=16,w2=16,d2=16) >0.1




with tqdm(range(len(real_abnormal_test))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(real_abnormal_test[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(real_abnormal_test[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            real_ab_test_list.append(ab_patch.reshape(-1))
            # print(ab_patch.shape)
        n_patch_mask = reduce(real_abnormal_test[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(real_abnormal_test[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        

        real_n_test_list.append(n_patches)
        pbar.update(0)

real_ab_test_list = np.stack(real_ab_test_list)
real_n_test_list = np.concatenate(real_n_test_list)
np.save('Patchestest_real_ab_test_brats_26_02_24_32.npy',real_ab_test_list)
np.save('Patchestest_real_n_test_brats_26_02_24_32.npy',real_n_test_list)


with tqdm(range(len(real_abnormal))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(real_abnormal[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(real_abnormal[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            real_ab_list.append(ab_patch.reshape(-1))

            # print(ab_patch.shape)
        n_patch_mask = reduce(real_abnormal[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(real_abnormal[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        
        real_n_list.append(n_patches)
        pbar.update(0)

real_ab_list = np.stack(real_ab_list)
real_n_list = np.concatenate(real_n_list)
np.save('Patchestest_real_ab_brats_26_02_24_32.npy',real_ab_list)
np.save('Patchestest_real_n_brats_26_02_24_32.npy',real_n_list)

with tqdm(range(len(sim_abnormal))) as pbar:
    for i in pbar:
        # real_ab_list.append(real_abnormal[i]['input'].reshape(-1))
        labels = label(sim_abnormal[i]['gt'][:,:,:,0])
        regions = regionprops(labels)
        ab_patches = []
        for region in regions:
            cent = (int(region.centroid[0]),int(region.centroid[1]),int(region.centroid[2]))
            #print(cent)
            padx1 = cent[0]-16
            if(padx1<0):
                padx1 = -padx1
            else:
                padx1 = 0
            padx2 = cent[0]+16-127
            if(padx2<0):
                padx2 = 0
            
            pady1 = cent[1]-16
            if(pady1<0):
                pady1 = -pady1
            else:
                pady1 = 0
            pady2 = cent[1]+16-127
            if(pady2<0):
                pady2 = 0

            padz1 = cent[2]-16
            if(padz1<0):
                padz1 = -padz1
            else:
                padz1 = 0
            padz2 = cent[2]+16-127
            if(padz2<0):
                padz2 = 0
            
            pad_shape = ((padx1,padx2),(pady1,pady2),(padz1,padz2))
            new_cent  = list(cent)
            new_cent[0]+=padx1-padx2
            new_cent[1]+=pady1-pady2
            new_cent[2]+=padz1-padz2

            image = np.pad(sim_abnormal[i]['input'][:,:,:,0],pad_shape)
            ab_patch = image[new_cent[0]-16:new_cent[0]+16,new_cent[1]-16:new_cent[1]+16,new_cent[2]-16:new_cent[2]+16]
            ab_patches.append(ab_patch.reshape(-1))

            sim_ab_list.append(ab_patch.reshape(-1))

            # print(ab_patch.shape)
        n_patch_mask = reduce(sim_abnormal[i]['gt'][:,:,:,0],'(h h2) (w w2) (d d2) -> (h w d)','mean',h2=32,w2=32,d2=32) <=0

        patched_image = rearrange(sim_abnormal[i]['input'],'(h h2) (w w2) (d d2) c-> (h w d) (c h2 w2 d2)',h2=32,w2=32,d2=32)

        n_patches = patched_image[n_patch_mask,:]
        sim_n_list.append(n_patches)
        pbar.update(0)

sim_ab_list = np.stack(sim_ab_list)
sim_n_list = np.concatenate(sim_n_list)
np.save('Patchestest_sim_ab_brats_26_02_24_32.npy',sim_ab_list)
np.save('Patchestest_sim_n_brats_26_02_24_32.npy',sim_n_list)





# print('Sizes are ',real_ab_test_list.shape, real_n_test_list.shape ,real_ab_list.shape, real_n_list.shape,sim_ab_list.shape,sim_n_list.shape)

"""

# test_real_ab = PCA(50).fit_transform(real_ab_list)
# test_real_n = PCA(50).fit_transform(real_n_list)

# test_real_ab_test = PCA(50).fit_transform(real_ab_test_list)
# test_real_n_test = PCA(50).fit_transform(real_n_test_list)

# test_sim_ab = PCA(50).fit_transform(sim_ab_list)
# test_sim_n = PCA(50).fit_transform(sim_n_list)

# np.random.shuffle(test_real_ab)
# np.random.shuffle(test_real_n)

# np.random.shuffle(test_real_ab_test)
# np.random.shuffle(test_real_n_test)

# np.random.shuffle(test_sim_ab)
# np.random.shuffle(test_sim_n)

# np.save('test_real_ab_brats_26_02_24_32.npy',test_real_ab)
# np.save('test_real_n_brats_26_02_24_32.npy',test_real_n)

# np.save('test_real_ab_test_brats_26_02_24_32.npy',test_real_ab_test)
# np.save('test_real_n_test_brats_26_02_24_32.npy',test_real_n_test)

# np.save('test_sim_ab_brats_26_02_24_32.npy',test_sim_ab)
# np.save('test_sim_n_brats_26_02_24_32.npy',test_sim_n)




test_real_ab = np.load('train_real_ab_brats_26_02_24_32.npy')
test_real_n = np.load('train_real_n_brats_26_02_24_32.npy')

test_real_ab_test = np.load('train_real_ab_test_brats_26_02_24_32.npy')
test_real_n_test = np.load('train_real_n_test_brats_26_02_24_32.npy')

test_sim_ab = np.load('train_sim_ab_brats_26_02_24_32.npy')
test_sim_n = np.load('train_sim_n_brats_26_02_24_32.npy')

print('Sizes are ', test_real_ab.shape, test_real_n.shape,test_sim_ab.shape,test_sim_n.shape)

min_val = min(test_real_ab_test.shape[0],test_real_n_test.shape[0],test_real_ab.shape[0],test_real_n.shape[0],test_sim_ab.shape[0],test_sim_n.shape[0])
print(min_val)

test_real_ab=test_real_ab[:min_val]
test_real_n = test_real_n[:min_val]
test_real_ab_test=test_real_ab_test[:min_val]
test_real_n_test = test_real_n_test[:min_val]
test_sim_ab = test_sim_ab[:min_val]
test_sim_n = test_sim_n[:min_val]

metrics = ['euclidean',]# 'euclidean', 'canberra', 'hamming', 'sokalmichener', 'braycurtis', 'russellrao', 'dice', 'l2', 'yule', 'manhattan', 'nan_euclidean', 'jaccard', 'rogerstanimoto', 'minkowski', 'cityblock', 'correlation', 'sqeuclidean', 'mahalanobis', 'sokalsneath', 'cosine', 'l1', 'kulsinski', 'matching']
for perplex in [15,20,25]:
    for i in metrics:
        print(perplex,i)
        test_tsne =  TSNE(3,perplexity=perplex,n_iter=10000,metric=i).fit_transform(np.concatenate([test_real_ab,test_sim_ab,test_real_ab_test,test_real_n,test_sim_n,test_real_n_test]))
        size_tsne = min_val
        ax = plt.axes(projection ="3d")

        ax.scatter3D(test_tsne[:size_tsne,0],test_tsne[:size_tsne,1],test_tsne[:size_tsne,2])
        ax.scatter3D(test_tsne[size_tsne:size_tsne*2,0],test_tsne[size_tsne:size_tsne*2,1],test_tsne[size_tsne:size_tsne*2,2])
     
        ax.scatter3D(test_tsne[size_tsne*2:size_tsne*3,0],test_tsne[size_tsne*2:size_tsne*3,1],test_tsne[size_tsne*2:size_tsne*3,2])
        ax.scatter3D(test_tsne[size_tsne*3:size_tsne*4,0],test_tsne[size_tsne*3:size_tsne*4,1],test_tsne[size_tsne*3:size_tsne*4,2])
     
        ax.scatter3D(test_tsne[size_tsne*4:size_tsne*5,0],test_tsne[size_tsne*4:size_tsne*5,1],test_tsne[size_tsne*4:size_tsne*5,2])
        ax.scatter3D(test_tsne[size_tsne*5:,0],test_tsne[size_tsne*5:,1],test_tsne[size_tsne*5:,2])

        plt.title('Perplexity '+str(perplex)+' '+i)
        plt.legend(['Real Abnormal','Sim Abnormal','Real Abnormal Test','Real Normal','Real Normal Test','Sim Normal'])
        plt.savefig('./images/2_03_24/3d/'+str(perplex)+'/tsne_3imgs_'+i+'.png')
        plt.show()
        plt.close()


"""