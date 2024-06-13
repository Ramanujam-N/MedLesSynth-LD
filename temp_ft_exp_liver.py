import numpy as np
import torch
import os
from monai.networks.nets import UNet 
model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )


model_dict = torch.load('./models/model_final.pt')
model.load_state_dict(model_dict['state_dict'])
model = model.cuda()

max_epochs = 200
for i in range(num_epochs):
