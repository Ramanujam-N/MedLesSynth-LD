MedLesSynth-LD : Lesion Synthesis using Physics-Based Noise
Models for Robust Lesion Segmentation in
Low-Data Medical Imaging Regimes

***

For creating new simulations 
python create_dataset_with_parser_args.py -data "include the setting which you wish to include (bratsonbrats)"

***

For training  
python train_with_parser_args.py -mode "setting" -data " data (brats/wmh/lits/busi)" -simulation_path " path for simulations" 

***

For testing  
python test_with_parser_args.py -mode "setting" -data " data (brats/wmh/lits/busi) " -pretrained " epoch number of best loss"

***