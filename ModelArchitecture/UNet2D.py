import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

# UNet2D
class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(UNet2D, self).__init__()
        features = init_features
        self.encoder1 = UNet2D._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet2D._block(features * 2, features, name="dec1")

        self.seg = nn.Conv2d( in_channels=features, out_channels=out_channels, kernel_size=1)
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        seg = self.seg(dec1)
        return self.outsig(seg)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

# UNet++

class NestedUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features = 16, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [init_features, init_features*2, init_features*4, init_features*8, init_features*16]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = torch.sigmoid(self.final(x0_4))    
            return output

# ResUNet2D

class ResConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.bnorm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels,out_channels,1,padding = 'same')

        self.bnorm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding = 'same')

        self.bnorm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels,out_channels,1,padding = 'same')

        # For shortcut 
        self.conv0 = nn.Conv2d(in_channels,out_channels,1,padding = 'same')
        self.bnorm0 = nn.BatchNorm2d(out_channels)

    def forward(self,x):
        x_init = x

        out = self.bnorm1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bnorm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bnorm3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        skip_connection = self.conv0(x_init)
        skip_connection = self.bnorm0(skip_connection)

        out = out + skip_connection
        return out


class ResUNet2D(nn.Module):
    def __init__(self,in_channels=1,out_channels=1,init_features=16):
        super().__init__()
        channels = [init_features, init_features*2, init_features*4, init_features*8, init_features*16]
        
        # Encoder
        self.encoder1 = ResConvBlock(in_channels,channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.encoder2 = ResConvBlock(channels[0],channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size =2,stride = 2)
        self.encoder3 = ResConvBlock(channels[1],channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size =2,stride = 2)
        self.encoder4 = ResConvBlock(channels[2],channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size =2,stride = 2)

        # Bridge
        self.bridge1 = ResConvBlock(channels[3],channels[4])
        self.bridge2 = ResConvBlock(channels[4],channels[4])

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = ResConvBlock(channels[3]*2,channels[3])
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = ResConvBlock(channels[2]*2,channels[2])
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = ResConvBlock(channels[1]*2,channels[1])
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = ResConvBlock(channels[0]*2,channels[0])

        # Output
        self.outputconv = nn.Conv2d(channels[0],out_channels,1,padding='same')
        self.outputnorm = nn.BatchNorm2d(out_channels)
        self.outputact = nn.Sigmoid()

    def forward(self,x):

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck1 = self.bridge1(self.pool4(enc4))
        bottleneck2 = self.bridge2(bottleneck1)


        dec4 = self.upconv4(bottleneck2)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        seg = self.outputconv(dec1)
        seg = self.outputnorm(seg)
        seg = self.outputact(seg)
        return seg


# Half UNet2D
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=bias, padding=padding)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class GhostModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel,pad):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels//2, kernel, padding = pad)
        self.batch1 = nn.BatchNorm2d(out_channels//2)
        self.act1 = nn.ReLU()
        self.conv2 = SeparableConv2d(out_channels//2,out_channels//2, kernel, padding = pad)
        self.batch2 = nn.BatchNorm2d(out_channels//2)
        self.act2 = nn.ReLU()
    def forward(self,x):
        out = self.conv1(x)
        out = self.batch1(out)
        out_act1 = self.act1(out)

        out = self.conv2(out_act1)
        out = self.batch2(out)
        out_act2 = self.act2(out)
        return torch.cat([out_act1, out_act2], dim = 1)

class HalfUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(HalfUNet2D, self).__init__()

        features = init_features
        self.encoder1 = HalfUNet2D._block(in_channels,features,name = 'enc1')
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = HalfUNet2D._block(features,features,name = 'enc2')
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = HalfUNet2D._block(features,features,name = 'enc3')
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = HalfUNet2D._block(features,features,name = 'enc4')
        self.pool4 = nn.MaxPool2d(2)
        # self.encoder5 = HalfUNet2D._block(features,features,name = 'enc5')
        # self.pool5 = nn.MaxPool2d(2)

        self.upsample2 =  nn.Upsample(scale_factor=2,mode='bilinear')
        self.upsample3 =  nn.Upsample(scale_factor=4,mode='bilinear')
        self.upsample4 =  nn.Upsample(scale_factor=8,mode='bilinear')
        # self.upsample5 =  nn.Upsample(scale_factor=16,mode='bilinear')

        self.encoderfinal = HalfUNet2D._block(features,features,name = 'encf')
        self.convout_1 = nn.Conv2d(features,out_channels,kernel_size = 1, padding='same')
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        # enc5 = self.encoder5(self.pool3(enc4))        


        # add_all = self.upsample5(enc5) + self.upsample4(enc4) + self.upsample3(enc3) + self.upsample2(enc2) + enc1 
        add_all = self.upsample4(enc4) + self.upsample3(enc3) + self.upsample2(enc2) + enc1
        out = self.encoderfinal(add_all)
        out = self.convout_1(out)
        out = self.sigmoid(out)
       
        return out
    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "ghost_module_1",
                        GhostModule(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel=3,
                            pad = 'same'
                        ),
                    ),

                    (
                        name + "ghost_module_2",
                        GhostModule(
                            in_channels=features,
                            out_channels=features,
                            kernel=3,
                            pad = 'same'
                        ),
                    ),
                ]
            )
        )
    

# SCAU Spatial and Channel Attention UNet2D

class SpatialAttention(nn.Module):
    def __init__(self, ):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding='same', bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spa = torch.cat([avg_out, max_out], dim=1)
        x_spa = self.conv1(x_spa)
        x_spa = self.sigmoid(x_spa)
        x = x*x_spa
        return x
    
class ChannelAttention(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelAttention, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    
    def forward(self, x):
        avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_sum = self.mlp( avg_pool )

        max_pool = F.max_pool2d( x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_sum += self.mlp( max_pool )

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        return x * scale
    



class SA_UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(SA_UNet2D, self).__init__()
        features = init_features
        self.encoder1 = SA_UNet2D._block(in_channels, features, name="enc1")
        self.sa_unit1 = SpatialAttention()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = SA_UNet2D._block(features, features * 2, name="enc2")
        self.sa_unit2 = SpatialAttention()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = SA_UNet2D._block(features * 2, features * 4, name="enc3")
        self.sa_unit3 = SpatialAttention()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = SA_UNet2D._block(features * 4, features * 8, name="enc4")
        self.sa_unit4 = SpatialAttention()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SA_UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = SA_UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = SA_UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = SA_UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = SA_UNet2D._block(features * 2, features, name="dec1")

        self.seg = nn.Conv2d( in_channels=features, out_channels=out_channels, kernel_size=1)
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.sa_unit4(enc4)), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.sa_unit3(enc3)), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.sa_unit2(enc2)), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.sa_unit1(enc1)), dim=1)
        dec1 = self.decoder1(dec1)
        seg = self.seg(dec1)
        return self.outsig(seg)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop1", nn.Dropout2d(p=0.15,inplace=True)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop2", nn.Dropout2d(p=0.15,inplace=True)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )
    
# CBAM unit with CHANNEL -> SPATIAL attention
class SAC_UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super(SAC_UNet2D, self).__init__()
        features = init_features
        self.encoder1 = SAC_UNet2D._block(in_channels, features, name="enc1")
        self.ca_unit1 = ChannelAttention(features)
        self.sa_unit1 = SpatialAttention()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = SAC_UNet2D._block(features, features * 2, name="enc2")
        self.ca_unit2 = ChannelAttention(features * 2)
        self.sa_unit2 = SpatialAttention()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = SAC_UNet2D._block(features * 2, features * 4, name="enc3")
        self.ca_unit3 = ChannelAttention(features * 4)
        self.sa_unit3 = SpatialAttention()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = SAC_UNet2D._block(features * 4, features * 8, name="enc4")
        self.ca_unit4 = ChannelAttention(features * 8)
        self.sa_unit4 = SpatialAttention()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = SAC_UNet2D._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = SAC_UNet2D._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = SAC_UNet2D._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = SAC_UNet2D._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = SAC_UNet2D._block(features * 2, features, name="dec1")

        self.seg = nn.Conv2d( in_channels=features, out_channels=out_channels, kernel_size=1)
        self.outsig = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, self.sa_unit4(self.ca_unit4(enc4))), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, self.sa_unit3(self.ca_unit3(enc3))), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, self.sa_unit2(self.ca_unit2(enc2))), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, self.sa_unit1(self.ca_unit1(enc1))), dim=1)
        dec1 = self.decoder1(dec1)
        seg = self.seg(dec1)
        return self.outsig(seg)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop1", nn.Dropout2d(p=0.15,inplace=True)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "drop2", nn.Dropout2d(p=0.15,inplace=True)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                ]
            )
        )