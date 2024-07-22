import torch
import torch.nn as nn
import argparse
import ast
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import pandas as pd
from helper_datadict import *
from tqdm import tqdm
from ModelArchitecture.metrics import *

CURRENT_DIRECTORY = os.getcwd()

MODEL_DIR = CURRENT_DIRECTORY + '/models/' 
RESULTS_DIR = CURRENT_DIRECTORY + '/results/' 

def save_image(i,image,label,label_wothresh,gndtruth,affine,dice,data,mode,exp_id,date):
    os.makedirs(f'./images/{date}/{mode}/{data}/{exp_id}/',exist_ok=True)
    if(affine==None):
        plt.imsave(f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_image.png',image,cmap='gray')
        plt.imsave(f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_outmask.png',label,cmap='gray')
        plt.imsave(f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_outmask_wothresh.png',label_wothresh,cmap='gray')
        plt.imsave(f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_manualmask.png',gndtruth,cmap='gray')

    else:
        image1 = nib.Nifti1Image(image, affine=affine)
        nib.save(image1, f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_FLAIR.nii.gz')

        output1 = nib.Nifti1Image(label.astype(np.single), affine=affine)
        nib.save(output1, f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_outmask.nii.gz')

        output2 = nib.Nifti1Image(label_wothresh.astype(np.single), affine=affine)
        nib.save(output2, f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_outmask_wothresh.nii.gz')

        output3 = nib.Nifti1Image(gndtruth.astype(np.single), affine=affine)
        nib.save(output3, f'./images/{date}/{mode}/{data}/{exp_id}/Testing{str(i)}_{str(dice)}_manualmask.nii.gz')


def test(system_data_path,mode,date,model,data,criterion,workers=4,batch=8,factor=1,device=0,pretrained=None,scale_factor =1,hyper_parameters=None,exp_id='default',sim_path=None,optthresh=False,size=(128,128,128),no_crop=False,combined_check="brats",not_best=False,thresh=0.5):
    model_save_name = MODEL_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion
    result_path = RESULTS_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion+'_loss' + '.npy'
    model_path = model_save_name + '_state_dict_best_dice' + str(pretrained) + '.pth'
    if(not_best==True):
        model_path = model_save_name + '_state_dict' + str(pretrained) + '.pth'
    which_data = data
    model_name = model
    _,_,datadict_test = helper_supervised(system_data_path,which_data=data,size=size,no_crop=no_crop)
    if(type(datadict_test)==tuple):
        if(combined_check=="brats"):
            datadict_test = datadict_test[0]
        else:
            datadict_test = datadict_test[1]
    testloader = DataLoader(datadict_test, batch_size=1, shuffle=False,num_workers=4)

    device = 'cuda:'+str(device)
    model = helper_model(model_type=model,which_data=data,hyper_parameters=hyper_parameters,device=device,size=size)
    model.load_state_dict(torch.load(model_path,map_location = device)['model_state_dict'])
    dice_list = []
    hau_list = []
    tpr_list = []
    tnr_list = []

    test_dice = 0
    test_hau_acc = 0
    test_tpr_acc = 0
    test_tnr_acc = 0

    # os.makedirs('./images/'+date+'/'+which_data,exist_ok =True)
    os.makedirs(f'./csv_files/{date}/{mode}/{data}/{exp_id}',exist_ok =True)

    model.eval()
    if(optthresh==False):
        with tqdm(range(len(testloader))) as pbar:
            for i, data in zip(pbar, testloader):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    err = 0
                    image = data['input'].to(device)
                    output = model.forward(image)
                    if(model_name =='unet_synth_tum'):
                        output = torch.sigmoid(output[:,2:,:,:,:])
                    output_wothresh = output
                    output = output>thresh

                    if(no_crop):
                        label = data['orig'].cpu().numpy().squeeze()
                        pred = output.cpu().numpy().squeeze()
                    else:
                        image,pred,pred_wothresh,label = helper_resize(image,output,output_wothresh,data['orig'],shape=data['shape'],crop_para=data['crop_para'])
                                    
                    dice = Dice_Score(pred,label)
                    hau_acc = Hausdorff_score(pred,label)
                    tpr_acc = TPR(pred,label)
                    tnr_acc = TNR(pred,label)
                    
                    pbar.set_postfix(
                                    Test_dice =  np.round(dice, 5), 
                                    Test_hau_acc = np.round(hau_acc, 5),Test_TPR= np.round(tpr_acc, 5),Test_TNR= np.round(tnr_acc, 5),)
                    pbar.update(0)
                    if('affine' not in data.keys()):
                        save_image(i,image,pred,pred_wothresh,label,None,np.round(dice, 2),which_data,mode,exp_id,date)
                    else:
                        save_image(i,image,pred,pred_wothresh,label,data['affine'][0],np.round(dice, 2),which_data,mode,exp_id,date)

                    test_dice += dice.item()
                    test_hau_acc += hau_acc.item()
                    test_tpr_acc += tpr_acc.item()
                    test_tnr_acc += tnr_acc.item()
                    dice_list.append(dice.item())
                    hau_list.append(hau_acc.item())
                    tpr_list.append(tpr_acc.item())
                    tnr_list.append(tnr_acc.item())

                    del image
                    del label
                    del err
        print(f'Dice {np.round(np.mean(dice_list),5)}({np.round(np.std(dice_list),5)})',
            f'Hausdorff {np.round(np.mean(hau_list),5)}({np.round(np.std(hau_list),5)})',
            f'TPR {np.round(np.mean(tpr_list),5)}({np.round(np.std(tpr_list),5)})',
            f'TNR {np.round(np.mean(tnr_list),5)}({np.round(np.std(tnr_list),5)})')
        pd.DataFrame({'Dice':dice_list,'Hausdorff':hau_list,'TPR':tpr_list,'TNR':tnr_list,}).to_csv(f'./csv_files/{date}/{mode}/{which_data}/{exp_id}/'+model_name+'_'+which_data+'_'+mode+'_'+exp_id+'_'+date+'_'+str(pretrained)+str(thresh)+'.csv')

    
    else:
        dice_thresholds=[]
        thresholds = list(np.arange(0,1.1,0.1))
        dice_list = {}
        hau_list = {}
        tpr_list = {}
        tnr_list = {}
        for i in range(4):
            dice_list = {str(thresh):[] for thresh in thresholds}
            hau_list = {str(thresh):[] for thresh in thresholds}
            tpr_list = {str(thresh):[] for thresh in thresholds}
            tnr_list = {str(thresh):[] for thresh in thresholds}

            test_dice = {str(thresh):0 for thresh in thresholds}
            test_hau_acc = {str(thresh):0 for thresh in thresholds}
            test_tpr_acc = {str(thresh):0 for thresh in thresholds}
            test_tnr_acc = {str(thresh):0 for thresh in thresholds}
        with tqdm(range(len(testloader))) as pbar:
            for i, data in zip(pbar, testloader):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    err = 0
                    image = data['input'].to(device)
                    output = model.forward(image)
                    output_wothresh = output
                    if(model_name =='unet_synth_tum'):
                        output = torch.sigmoid(output[:,2:,:,:,:])
                    for thresh in thresholds:
                        output = output_wothresh>thresh
                        # image,pred,label = helper_resize(image,output,data['orig'],shape=data['shape'],crop_para=data['crop_para'])
                        if(no_crop):
                            label = data['orig'].cpu().numpy().squeeze()
                            pred = output.cpu().numpy().squeeze()
                        else:
                            image_cpu,pred,pred_wothresh,label = helper_resize(image,output,output_wothresh,data['orig'],shape=data['shape'],crop_para=data['crop_para'])
                        

                        dice = Dice_Score(pred,label,threshold=thresh)
                        hau_acc = Hausdorff_score(pred,label,threshold=thresh)
                        tpr_acc = TPR(pred,label,threshold=thresh)
                        tnr_acc = TNR(pred,label,threshold=thresh)
                        
                        pbar.set_postfix(Thresh=thresh,
                                        Test_dice =  np.round(dice, 5), 
                                        Test_hau_acc = np.round(hau_acc, 5),Test_TPR= np.round(tpr_acc, 5),Test_TNR= np.round(tnr_acc, 5),)
                        pbar.update(0)
                        test_dice[str(thresh)] += dice.item()
                        test_hau_acc[str(thresh)] += hau_acc.item()
                        test_tpr_acc[str(thresh)] += tpr_acc.item()
                        test_tnr_acc[str(thresh)] += tnr_acc.item()
                        dice_list[str(thresh)].append(dice.item())
                        hau_list[str(thresh)].append(hau_acc.item())
                        tpr_list[str(thresh)].append(tpr_acc.item())
                        tnr_list[str(thresh)].append(tnr_acc.item())

                    del image
                    del image_cpu
                    del label
                    del err
        for i in range(len(thresholds)):
            dice_thresholds.append(np.round(np.mean(dice_list[str(thresh)]),5))
            print(f'For threshold {thresh} ',f'Dice {np.round(np.mean(dice_list[str(thresh)]),5)}({np.round(np.std(dice_list[str(thresh)]),5)})',
            f'Hausdorff {np.round(np.mean(hau_list[str(thresh)]),5)}({np.round(np.std(hau_list[str(thresh)]),5)})',
            f'TPR {np.round(np.mean(tpr_list[str(thresh)]),5)}({np.round(np.std(tpr_list[str(thresh)]),5)})',
            f'TNR {np.round(np.mean(tnr_list[str(thresh)]),5)}({np.round(np.std(tnr_list[str(thresh)]),5)})')
            pd.DataFrame({'Dice':dice_list[str(thresh)],'Hausdorff':hau_list[str(thresh)],'TPR':tpr_list[str(thresh)],'TNR':tnr_list[str(thresh)],}).to_csv(f'./csv_files/{date}/{mode}/{which_data}/{exp_id}/'+model_name+'_'+which_data+'_'+mode+'_'+exp_id+'_'+date+'_'+str(pretrained)+'_thresh_'+str(thresh)+'.csv')
        print('Best Threshold was {}'.format(thresholds[np.argmax(dice_thresholds)]))

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",default='S',choices=['S','SS','SSDA','DA','FT','PT','FTDA','FTSS','FPI'],
                        help="mode to run the model in")
    parser.add_argument("-data",default='brats',choices=['texshapes','randomshapes','spheres','combinedv2','combinedv1','wmh','brats','busi','lits','idrid'],help='Which data to run on?')
    parser.add_argument("-model",default='unet', choices=['unet_dropout','unet_synth_tum','unet','slimunetr','ducknet','saunet','nestedunet','halfunet','resunet','unetr','sacunet'],help='Which model to run ?')
    parser.add_argument("-loss",dest='criterion',default='focal + dice',choices=['dicece','dice','focal + dice'],help='Which loss to choose?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-device",default=0,type=int,choices=[0,1])
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-sim_factor",type=float,dest='factor',default=1.0,choices=[0.2,0.4,0.6,0.8,1.0,2.0,5.0])
    parser.add_argument("-real_factor",type=float,dest='scale_factor',default=1.0,choices=[0.2,0.4,0.6,0.75,1.0])
    parser.add_argument("-pretrained",type=int,dest='pretrained',help='Give self supervised pre trained models index to start fine tuning')
    parser.add_argument("-hyperparam",default="{'init_features':16}",dest='hyper_parameters',type=ast.literal_eval,help='Pass dictionary of hyperparameter if needs changing.')
    parser.add_argument("-exp_id",default='default',help='Name to uniquely identify the experiment')
    parser.add_argument("-simulation_path",dest='sim_path',help='Path to simulated files')
    parser.add_argument("-size",nargs='+', type=int,help='To run it in orginal dimensions')
    parser.add_argument("-combined_check",default='brats',choices = ['brats','wmh'],help='Which data to check in combined')
    parser.add_argument("-no_crop",default=False,action='store_true',help='To not have the model tight crop the images')    
    parser.add_argument("-optthresh",action='store_true')
    parser.add_argument("-data_path",dest='system_data_path',default=112,type=int,choices=[131,112,63,64])
    parser.add_argument("-not_best",action='store_true')
    parser.add_argument("-thresh",default=0.5,type=float,)
    args = parser.parse_args()

    mode_dir = {'FPI':'FPI','S':'Supervised','SS':'Self_Supervised','SSDA':'Self_Supervised_Data_Adaptation','SSDA_v2':'Self_Supervised_Data_Adaptation_v2','DA':'Data_Augmentation','FT':'Fine_Tuning','PT':'Pre_Training','FTDA':'Fine_Tuning_Data_Augmentation','FTSS':'Fine_Tuning_Self_Supervised'}
    args.mode = mode_dir[args.mode]
    data_addresses = {63:'/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6',64:'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a',65:'/mnt/b0305b0a-824d-48cb-a829-2a6766e6b45b',112:'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f',131:'/mnt/a4ef64ea-1b6b-4423-b1d2-4794d2e97289'}
    args.system_data_path = data_addresses[args.system_data_path]

    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    test(**vars(args))