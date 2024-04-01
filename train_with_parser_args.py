import torch
import torch.nn as nn
import argparse
import ast
from datetime import datetime
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from helper_datadict import *
from tqdm import tqdm

CURRENT_DIRECTORY = os.getcwd()

MODEL_DIR = CURRENT_DIRECTORY + '/models/' 
RESULTS_DIR = CURRENT_DIRECTORY + '/results/' 

def train(mode,date,criterion,model,data,workers=4,batch=8,factor=1,device=0,checkpoint=None,pretrained=None,scale_factor =1,hyper_parameters=None,no_aug=False,exp_id='default',sim_path=None,size=(128,128,128),no_crop=False,system_data_path=112):
    model_save_name = MODEL_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion
    result_save_name = RESULTS_DIR + date+'/'+mode+'/'+data+'/'+exp_id+'/'+model+'_'+criterion
    checkpoint_path = model_save_name + '_state_dict_best_dice' + str(checkpoint) + '.pth'
    
    ss_path = MODEL_DIR + date+'/'+'Self_Supervised'+'/'+data+'/'+exp_id+'/'
    pt_path = MODEL_DIR + date+'/'+'Pre_Training'+'/'+data+'/'+exp_id+'/'
    fine_tuning_path = pt_path+model+'_'+criterion + '_state_dict_best_dice' + str(pretrained) +'.pth'
    ss_da_path = ss_path+model+'_'+criterion + '_state_dict_best_dice' + str(pretrained) +'.pth'
    ss_da_save_path = ss_path + str(pretrained) +'/'

    print(checkpoint_path)
    print(pt_path)
    os.makedirs(MODEL_DIR+date+'/'+mode+'/'+data+'/'+exp_id, exist_ok=True)
    os.makedirs(RESULTS_DIR+date+'/'+mode+'/'+data+'/'+exp_id, exist_ok=True)

    helper_transformation(no_aug)

    if(mode == 'Supervised'):
        datadict_train,datadict_val,datadict_test = helper_supervised(system_data_path,which_data=data,size=size,no_crop=no_crop)
    elif(mode == 'Self_Supervised'):
        datadict_train,datadict_val = helper_self_supervised(which_data=data,scale_factor=scale_factor,sim_path_other=sim_path,size=size,no_crop=no_crop)
    elif(mode == 'Pre_Training'):
        datadict_train,datadict_val = helper_pre_training(which_data=data,scale_factor=scale_factor,sim_path_other=sim_path,size=size,no_crop=no_crop)
    elif(mode == 'Data_Augmentation'):
        datadict_train,datadict_val = helper_data_augmentation(system_data_path,which_data=data,scale_factor=scale_factor,factor=factor,sim_path_other=sim_path,size=size,no_crop=no_crop)
    elif(mode == 'Fine_Tuning_Data_Augmentation'):
        datadict_train,datadict_val = helper_data_augmentation(system_data_path,which_data=data,scale_factor=scale_factor,factor=factor,sim_path_other=sim_path,size=size,no_crop=no_crop)
    elif(mode == 'Fine_Tuning'):
        datadict_train,datadict_val = helper_fine_tuning(system_data_path,which_data=data,factor=factor,size=size,no_crop=no_crop)
    elif(mode == 'Self_Supervised_Data_Adaptation'):
        datadict_train,datadict_val = helper_ss_data_adaptation(which_data=data,factor=factor,adapt_path = ss_da_path,adapt_save_path=ss_da_save_path, model=model, hyper_parameters=hyper_parameters, device=device)


    if(type(datadict_train)==tuple):
        trainloader1 = DataLoader(datadict_train[0], batch_size=batch, shuffle=True,num_workers=workers)
        valloader1 = DataLoader(datadict_val[0], batch_size=1, shuffle=False,num_workers=1)

        trainloader2 = DataLoader(datadict_train[1], batch_size=batch, shuffle=True,num_workers=workers)
        valloader2 = DataLoader(datadict_val[1], batch_size=1, shuffle=False,num_workers=1)
        trainloader = (trainloader1,trainloader2)
        valloader = (valloader1,valloader2)
    else:
        trainloader = DataLoader(datadict_train, batch_size=batch, shuffle=True,num_workers=workers)
        valloader = DataLoader(datadict_val, batch_size=1, shuffle=False,num_workers=1)
    
    device = 'cuda:'+str(device)

    model = helper_model(model_type=model,which_data=data,hyper_parameters=hyper_parameters,device=device,size=size)
    criterion = helper_criterion(criterion_type=criterion)
    
    if(fine_tuning_path!=None and (mode == 'Fine_Tuning_Data_Augmentation' or mode == 'Fine_Tuning')):
        print('----------------------------------Fine Tuning Path loaded----------------------------------')
        model.load_state_dict(torch.load(fine_tuning_path)['model_state_dict'])


    optimizer = optim.Adam(model.parameters(), lr = 1e-3, eps = 0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=10,min_lr = 1e-4,mode='max')

    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    best_loss = np.inf
    best_dice = -np.inf
    min_epoch = 0

    if(checkpoint!=None):
        checkpoint_dict = torch.load(checkpoint_path) 
        model.load_state_dict(checkpoint_dict['model_state_dict'])
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_dict['lr_scheduler_state_dict'])
        losses = np.load(result_save_name+'_loss' + '.npy')

        train_losses = losses[0,:].tolist()[:checkpoint_dict['epoch']+1]
        val_losses = losses[1,:].tolist()[:checkpoint_dict['epoch']+1]
        train_dices = losses[2,:].tolist()[:checkpoint_dict['epoch']+1]
        val_dices = losses[3,:].tolist()[:checkpoint_dict['epoch']+1]

        min_epoch = checkpoint_dict['epoch'] + 1
        best_loss = checkpoint_dict['loss']
        best_dice = checkpoint_dict['dice']
        print(best_loss,best_dice,len(train_losses),val_losses[-1][0]*len(valloader))

    num_epochs = 200


    for epoch in range(min_epoch,num_epochs):
        torch.cuda.empty_cache()
        
        epoch_loss,epoch_dice = helper_train(epoch,model,criterion,optimizer,scheduler,train_losses,train_dices,val_losses=val_losses,val_dices=val_dices,trainloader=trainloader,valloader=valloader,device=device)
        if(mode=='Self_Supervised_Data_Adaptation_v2'):
            re_epoch_loss,re_epoch_dice = helper_train(epoch,model,criterion,optimizer,scheduler,train_losses,train_dices,val_losses=val_losses,val_dices=val_dices,trainloader=re_trainloader,valloader=re_valloader,adapt_model=model,device=device)
            epoch_loss+=re_epoch_loss
            epoch_dice+=re_epoch_dice

        if(epoch_dice>best_dice):
                best_dice = epoch_dice
                helper_save_model(epoch,model,optimizer,epoch_loss,epoch_dice,scheduler,model_save_name,best_dice=True)
                
        if(epoch%10==0):
            helper_save_model(epoch,model,optimizer,epoch_loss,epoch_dice,scheduler,model_save_name,)

        np.save(result_save_name+'_loss' + '.npy', [train_losses,val_losses,train_dices,val_dices])

    helper_save_model(epoch,model,optimizer,epoch_loss,epoch_dice,scheduler,model_save_name)


if(__name__ =="__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode",default='S',choices=['S','SS','SSDA','DA','FT','FTDA','PT','FPI'],
                        help="mode to run the model in")
    parser.add_argument("-data",default='brats',choices=['total3d','idrid','texshapes','randomshapes','spheres','combinedv1','wmh','brats','busi','lits'],help='Which data to run on?')
    parser.add_argument("-model",default='unet', choices=['unet','slimunetr','ducknet','saunet','nestedunet','halfunet','resunet','unetr','sacunet'],help='Which model to run ?')
    parser.add_argument("-loss",dest='criterion',default='focal + dice',choices=['dice','focal + dice'],help='Which loss to choose?')
    parser.add_argument("-workers",default=4,type=int)
    parser.add_argument("-device",default=0,type=int,choices=[0,1])
    parser.add_argument("-batch",default=8,type=int)
    parser.add_argument("-date",default="{:%d_%m_%y}".format(datetime.now()))
    parser.add_argument("-sim_factor",type=float,dest='scale_factor',default=1.0,choices=[0.2,0.4,0.6,0.75,1.0,2.0,5.0])
    parser.add_argument("-real_factor",type=float,dest='factor',default=1.0,choices=[0.2,0.4,0.6,0.75,1.0])
    parser.add_argument("-checkpoint",type=int,help='Give checkpoint index to start model training from')
    parser.add_argument("-pretrained",type=int,dest='pretrained',help='Give self supervised pre trained models index to start fine tuning')
    parser.add_argument("-hyperparam",default="{'init_features':16}",dest='hyper_parameters',type=ast.literal_eval,help='Pass dictionary of hyperparameter if needs changing.')
    parser.add_argument("-no_aug",action='store_true')
    parser.add_argument("-exp_id",default='default',help='Name to uniquely identify the experiment')
    parser.add_argument("-simulation_path",dest='sim_path',help='Path to simulated files')
    parser.add_argument("-size",nargs='+',default=(128,128,128), type=int,help='To run it in orginal dimensions')
    parser.add_argument("-no_crop",default=False,action='store_true',help='To not have the model tight crop the images')
    parser.add_argument("-data_path",dest='system_data_path',default=112,type=int,choices=[112,63,64])
    args = parser.parse_args()

    mode_dir = {'FTDA':'Fine_Tuning_Data_Augmentation','S':'Supervised','SS':'Self_Supervised','SSDA':'Self_Supervised_Data_Adaptation','SSDA_v2':'Self_Supervised_Data_Adaptation_v2','DA':'Data_Augmentation','FT':'Fine_Tuning','PT':'Pre_Training'}
    data_addresses = {63:'/mnt/d1bdf387-8fd2-4f57-8c8a-eba9ef0baff6',64:'/mnt/70b9cd2d-ce8a-4b10-bb6d-96ae6a51130a',112:'/mnt/04d05e02-a59c-4a91-8c16-28a8c9f1c14f',}
    args.system_data_path = data_addresses[args.system_data_path]
    args.mode = mode_dir[args.mode]
    
    print("-----------------------------Arguments for the current execution-----------------------------------")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    train(**vars(args))