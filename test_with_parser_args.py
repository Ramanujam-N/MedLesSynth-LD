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

def save_image(i,image,label,gndtruth,affine,dice,combined_check,date):
    image1 = nib.Nifti1Image(image, affine=affine)
    nib.save(image1, './images/'+date+'/'+combined_check+'/'+'Testing'+str(i)+str(dice)+'_FLAIR.nii.gz')

    output1 = nib.Nifti1Image(label.astype(np.single), affine=affine)
    nib.save(output1, './images/'+date+'/'+combined_check+'/'+'Testing'+str(i)+str(dice)+'_outmask.nii.gz')

    output2 = nib.Nifti1Image(gndtruth.astype(np.single), affine=affine)
    nib.save(output2, './images/'+date+'/'+combined_check+'/'+'Testing'+str(i)+str(dice)+'_manualmask.nii.gz')


def test(mode,date,model,data,criterion,workers=4,batch=8,factor=1,device=0,pretrained=None,scale_factor =1,hyper_parameters=None,exp_id='default',sim_path=None,optthresh=False,size=(128,128,128),no_crop=False,combined_check="brats"):
    model_save_name = MODEL_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion
    result_path = RESULTS_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion+'_loss' + '.npy'
    model_path = model_save_name + '_state_dict_best_dice' + str(pretrained) + '.pth'
    which_data = data
    model_name = model
    _,_,datadict_test = helper_supervised(which_data=data,size=size,no_crop=no_crop)
    if(type(datadict_test)==tuple):
        if(combined_check=="brats"):
            datadict_test = datadict_test[0]
        else:
            datadict_test = datadict_test[1]
    testloader = DataLoader(datadict_test, batch_size=1, shuffle=False,num_workers=1)

    device = 'cuda:'+str(device)
    model = helper_model(model_type=model,which_data=data,hyper_parameters=hyper_parameters,device=device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    dice_list = []
    f1_list = []
    tpr_list = []
    tnr_list = []

    dice_list_70 = []
    tpr_list_70 = []

    dice_list_50 = []
    tpr_list_50 = []


    test_dice = 0
    test_f1_acc = 0
    test_tpr_acc = 0
    test_tnr_acc = 0

    model.eval()
    with tqdm(range(len(testloader))) as pbar:
        for i, data in zip(pbar, testloader):
            with torch.no_grad():
                torch.cuda.empty_cache()
                err = 0
                image = data['input'].to(device)
                output = model.forward(image)
                output = output>0.5

                if(no_crop):
                    label = data['orig'].cpu().numpy().squeeze()
                    pred = output.cpu().numpy().squeeze()
                else:
                    image,pred,label = helper_resize(image,output,data['orig'],shape=data['shape'],crop_para=data['crop_para'])

                dice = Dice_Score(pred,label)
                f1_acc = F1_score(pred,label)
                tpr_acc = TPR(pred,label)
                tnr_acc = TNR(pred,label)
                
                pbar.set_postfix(
                                Test_dice =  np.round(dice, 5), 
                                Test_f1_acc = np.round(f1_acc, 5),Test_TPR= np.round(tpr_acc, 5),Test_TNR= np.round(tnr_acc, 5),)
                pbar.update(0)

                test_dice += dice.item()
                test_f1_acc += f1_acc.item()
                test_tpr_acc += tpr_acc.item()
                test_tnr_acc += tnr_acc.item()
                dice_list.append(dice.item())
                f1_list.append(f1_acc.item())
                tpr_list.append(tpr_acc.item())
                tnr_list.append(tnr_acc.item())

                if(dice>0.7):
                    dice_list_70.append(dice.item())
                    tpr_list_70.append(tpr_acc.item())

                if(dice>0.5):
                    dice_list_50.append(dice.item())
                    tpr_list_50.append(tpr_acc.item())

                del image
                del label
                del err
    print(f'Dice {np.round(np.mean(dice_list),5)}({np.round(np.std(dice_list),5)})',
          f'F1 {np.round(np.mean(f1_list),5)}({np.round(np.std(f1_list),5)})',
          f'TPR {np.round(np.mean(tpr_list),5)}({np.round(np.std(tpr_list),5)})',
          f'TNR {np.round(np.mean(tnr_list),5)}({np.round(np.std(tnr_list),5)})')
    pd.DataFrame({'Dice':dice_list,'TPR':tpr_list}).to_csv(model_name+combined_check+which_data+mode+exp_id+date+'.csv')

    if(optthresh):
        dice_thresholds=[]
        thresholds = list(np.arange(0.1,0.99,0.1))
        for thresh in thresholds:
            dice_list = []
            f1_list = []
            tpr_list = []
            tnr_list = []
            test_dice = 0
            test_f1_acc = 0
            test_tpr_acc = 0
            test_tnr_acc = 0
            for i, data in zip(pbar, testloader):
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    err = 0
                    image = data['input'].to(device)
                    output = model.forward(image)
                    output = output>thresh
                    image,pred,label = helper_resize(image,output,data['orig'],shape=data['shape'],crop_para=data['crop_para'])
                    

                    dice = Dice_Score(pred,label,threshold=thresh)
                    f1_acc = F1_score(pred,label,threshold=thresh)
                    tpr_acc = TPR(pred,label,threshold=thresh)
                    tnr_acc = TNR(pred,label,threshold=thresh)
                    
                    pbar.set_postfix(
                                    Test_dice =  np.round(dice, 5), 
                                    Test_f1_acc = np.round(f1_acc, 5),Test_TPR= np.round(tpr_acc, 5),Test_TNR= np.round(tnr_acc, 5),)
                    pbar.update(0)
                    test_dice += dice.item()
                    test_f1_acc += f1_acc.item()
                    test_tpr_acc += tpr_acc.item()
                    test_tnr_acc += tnr_acc.item()
                    dice_list.append(dice.item())
                    f1_list.append(f1_acc.item())
                    tpr_list.append(tpr_acc.item())
                    tnr_list.append(tnr_acc.item())

                    del image
                    del label
                    del err
            dice_thresholds.append(np.round(np.mean(dice_list),5))
            print(f'For threshold {thresh} ',f'Dice {np.round(np.mean(dice_list),5)}({np.round(np.std(dice_list),5)})',
          f'F1 {np.round(np.mean(f1_list),5)}({np.round(np.std(f1_list),5)})',
          f'TPR {np.round(np.mean(tpr_list),5)}({np.round(np.std(tpr_list),5)})',
          f'TNR {np.round(np.mean(tnr_list),5)}({np.round(np.std(tnr_list),5)})')
        print('Best Threshold was {}'.format(thresholds[np.argmax(dice_thresholds)]))

if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",default='S',choices=['S','SS','SSDA','DA','FT','FPI'],
                        help="mode to run the model in")
    parser.add_argument("-data",default='brats',choices=['texshapes','randomshapes','spheres','combinedv2','combinedv1','wmh','brats','busi','lits','stare'],help='Which data to run on?')
    parser.add_argument("-model",default='unet', choices=['unet','slimunetr','ducknet','saunet','nestedunet','halfunet','resunet','unetr','sacunet'],help='Which model to run ?')
    parser.add_argument("-loss",dest='criterion',default='focal + dice',choices=['dice','focal + dice'],help='Which loss to choose?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-device",default=0,type=int,choices=[0,1])
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-sim_factor",type=float,dest='factor',default=1.0,choices=[0.2,0.4,0.6,0.8,1.0])
    parser.add_argument("-real_factor",type=float,dest='scale_factor',default=1.0,choices=[0.2,0.4,0.6,0.8,1.0])
    parser.add_argument("-pretrained",type=int,dest='pretrained',help='Give self supervised pre trained models index to start fine tuning')
    parser.add_argument("-hyperparam",default="{'init_features':16}",dest='hyper_parameters',type=ast.literal_eval,help='Pass dictionary of hyperparameter if needs changing.')
    parser.add_argument("-exp_id",default='default',help='Name to uniquely identify the experiment')
    parser.add_argument("-size",nargs='+', type=int,help='To run it in orginal dimensions')
    parser.add_argument("-combined_check",default='brats',choices = ['brats','wmh'],help='Which data to check in combined')
    parser.add_argument("-no_crop",default=False,action='store_true',help='To not have the model tight crop the images')    
    parser.add_argument("-optthresh",action='store_true')
    args = parser.parse_args()

    mode_dir = {'FPI':'FPI','S':'Supervised','SS':'Self_Supervised','SSDA':'Self_Supervised_Data_Adaptation','SSDA_v2':'Self_Supervised_Data_Adaptation_v2','DA':'Data_Augmentation','FT':'Fine_Tuning'}
    args.mode = mode_dir[args.mode]
    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    test(**vars(args))