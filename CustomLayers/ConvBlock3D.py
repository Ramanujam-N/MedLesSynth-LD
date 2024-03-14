import torch
import torch.nn as nn
import math
import torch.nn.functional as F
kernel_initializer = 'he_uniform'

class Gen_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters, block_type, dilation=1, size=3, repeat=1):
        super().__init__()
        self.block_type = block_type
        self.filters =  filters
        self.block_dict = {'seperated':Seperated_Conv3d_Block, 'duck':Duck_Conv3d_Block, 'midscope': Midscope_Conv3d_Block, 
                           'widescope':Widescope_Conv3d_Block, 'resnet':Resnet_Conv3d_Block,'double_convolution':DoubleConv_with_BatchNorm, 'conv':Conv3d_Block}
        blocks = [self.block_dict[self.block_type](in_filters,filters,dilation,size)] + [self.block_dict[self.block_type](filters,filters,dilation,size) for i in range(repeat-1)]
        self.conv_block = nn.Sequential(*blocks)

    def forward(self,x):
        out = self.conv_block(x)
        return out

class Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv = Conv3dSame(in_filters, filters, (size, size, size), )
    def forward(self,x):
        out = self.conv(x)

class Duck_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_filters)
        self.wide = Widescope_Conv3d_Block(in_filters,filters)
        self.mid = Midscope_Conv3d_Block(in_filters,filters)
        
        self.res_1 = Resnet_Conv3d_Block(in_filters,filters)
        
        self.res_2_1 = Resnet_Conv3d_Block(in_filters,filters)
        self.res_2_2 = Resnet_Conv3d_Block(filters,filters)

        self.res_3_1 = Resnet_Conv3d_Block(in_filters,filters)
        self.res_3_2 = Resnet_Conv3d_Block(filters,filters)
        self.res_3_3 = Resnet_Conv3d_Block(filters,filters)

        self.sep = Seperated_Conv3d_Block(in_filters,filters,size=6)
        
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.bn1(x)

        out_wide = self.wide(out)
        out_mid = self.mid(out)
        
        out_res_1 = self.res_1(out)
        
        out_res_2 = self.res_2_1(out)
        out_res_2 = self.res_2_2(out_res_2)

        out_res_3 = self.res_3_1(out)
        out_res_3 = self.res_3_2(out_res_3)
        out_res_3 = self.res_3_3(out_res_3)

        out_sep = self.sep(out)

        out  = out_wide + out_mid + out_res_1 + out_res_2 + out_res_3 + out_sep 
        #out  = out_mid + out_res_1 + out_res_2 + out_res_3 + out_sep

        out = self.bn2(out)
        
        return out

class Seperated_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv1 = Conv3dSame(in_filters, filters, (1, 1, size), )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = Conv3dSame(filters, filters, (1, size, 1), )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv3 = Conv3dSame(filters, filters, (size, 1, 1), )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn3(out)

        return out        

class Midscope_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv1 = Conv3dSame(in_filters, filters, (3, 3, 3), dilation=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = Conv3dSame(filters, filters, (3, 3, 3),  dilation=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        return out        

class Widescope_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv1 = Conv3dSame(in_filters, filters, (3, 3, 3), dilation=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)

        self.conv2 = Conv3dSame(filters, filters, (3, 3, 3),  dilation=2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)

        self.conv3 = Conv3dSame(filters, filters, (3, 3, 3),  dilation=3)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.bn3(out)
        return out        

class Resnet_Conv3d_Block(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv_res = Conv3dSame(in_filters, filters, (1, 1, 1),  
               dilation=dilation_rate)
        
        self.conv1 = Conv3dSame(in_filters, filters, (3, 3, 3),
               dilation=dilation_rate)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)  
        self.conv2 = Conv3dSame(filters, filters, (3, 3, 3), 
               dilation=dilation_rate)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)  

        self.bn3 = nn.BatchNorm3d(filters)

    def forward(self,x):
        out_res = self.conv_res(x)

        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        
        out = out + out_res
        out = self.bn3(out)

        return out        

class DoubleConv_with_BatchNorm(nn.Module):
    def __init__(self,in_filters,filters,size=3,dilation_rate=1):
        super().__init__()
        self.conv1 = Conv3dSame(in_filters, filters, (3, 3, 3),
               dilation=dilation_rate)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm3d(filters)  
        self.conv2 = Conv3dSame(filters, filters, (3, 3, 3),
               dilation=dilation_rate)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm3d(filters)  

    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.bn1(out)
                         
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.bn2(out)
        return out        




class Conv3dSame(torch.nn.Conv3d):

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw, id = x.size()[-3:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=id, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])


        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(
                x, [pad_d // 2, pad_d - pad_d // 2,pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

# conv_layer_s2_same = Conv3dSame(in_channels=3, out_channels=64, kernel_size=(7, 7, 7), stride=(2, 2, 2), groups=1, bias=True)
# out = conv_layer_s2_same(torch.zeros(1, 3, 224, 224, 224))
# print(out.shape)