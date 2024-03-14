import torch
import torch.nn as nn
from collections import OrderedDict

class ResNet3DBasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, expansion=1, downsample=None):
        super().__init__()
        self.expansion = expansion
        self.downsample = downsample

        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(output_channels, output_channels*self.expansion, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(output_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
    
class ResNet3D(nn.Module):

    def __init__(self, image_channels,attention_mode = True):
        super().__init__()
        self.expansion = 1
        self.in_channels = 64

        self.conv1 = nn.Conv3d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResNet3DBasicBlock,64, 64, 2, stride=1)
        self.layer2 = self._make_layer(ResNet3DBasicBlock,64, 128, 2, stride=2)
        self.layer3 = self._make_layer(ResNet3DBasicBlock,128, 256, 2, stride=2)
        self.layer4 = self._make_layer(ResNet3DBasicBlock,256, 512, 2, stride=2)
        # self.layer5 = self._make_layer(ResNet3DBasicBlock,512, 1024, 2, stride=2)
        # self.layer6 = self._make_layer(ResNet3DBasicBlock,1024, 2048, 2, stride=2)
        
        self.attention_mode = attention_mode
        if(self.attention_mode):
            self.upsample_layer1 = nn.Upsample(scale_factor = 2*2,mode='trilinear')
            self.conv1x1_layer1 = nn.Conv3d(64,1,kernel_size=1)

            # self.upsample_layer2 = nn.Upsample(scale_factor = 4*2,mode='trilinear')
            # self.conv1x1_layer2 = nn.Conv3d(128,1,kernel_size=1)

            # self.upsample_layer3 = nn.Upsample(scale_factor = 8*2,mode='trilinear')
            # self.conv1x1_layer3 = nn.Conv3d(256,1,kernel_size=1)

            # self.upsample_layer4 = nn.Upsample(scale_factor = 16*2,mode='trilinear')
            # self.conv1x1_layer4 = nn.Conv3d(512,1,kernel_size=1)

            # self.conv1x1_final = nn.Conv3d(4,1,kernel_size=1)

    def _make_layer(self, block,in_channels, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, output_channels*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm3d(output_channels*self.expansion),
            )
        
        layers = []
        layers.append(
            ResNet3DBasicBlock(
                self.in_channels,
                output_channels, 
                stride,
                self.expansion,
                downsample
            )
        )
        self.in_channels = output_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(
                ResNet3DBasicBlock(
                    self.in_channels,
                    output_channels,
                    expansion=self.expansion
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):

        if(self.attention_mode==False):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            # x = self.layer5(x)
            # x = self.layer6(x)

            return x
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            up_out1 = self.conv1x1_layer1(self.upsample_layer1(x))
            x = self.layer2(x)
            #up_out2 = self.conv1x1_layer2(self.upsample_layer2(x))
            x = self.layer3(x)
            #up_out3 = self.conv1x1_layer3(self.upsample_layer3(x))
            x = self.layer4(x)
            #up_out4 = self.conv1x1_layer4(self.upsample_layer4(x))

            #out_conv = self.conv1x1_final(torch.cat([up_out1,up_out2,up_out3,up_out4],dim=1))

            return x,up_out1
    


class VGG_Encoder(nn.Module):
    def __init__(self, in_channels=1, out_classes=1, init_features=64):
        super(VGG_Encoder, self).__init__()
        features = init_features
        self.encoder1 = VGG_Encoder._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = VGG_Encoder._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = VGG_Encoder._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder4 = VGG_Encoder._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.bottleneck = VGG_Encoder._block(features * 8, features * 16, name="bottleneck")

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))
        return (enc1,enc2,enc3,enc4,bottleneck)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.BatchNorm3d(num_features=features)),

                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.BatchNorm3d(num_features=features)),
                ]
            )
        )






class VGG3D(nn.Module):

    def __init__(self, input_channels,):
        super().__init__()

        self.enc_layer1 = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.enc_layer2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

        self.enc_layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.enc_layer4 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer5 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.enc_layer6 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True)
        )

        self.enc_layer7 = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer8 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.enc_layer9 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True)
        )

        self.enc_layer10 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)
        )

        self.enc_layer11 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2) # Newly added
        )

        self.enc_layer12 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size = 2, stride = 2)# Newly added
        )

        self.proj_layer1 = nn.Sequential(
            nn.Linear(4*4*4*512, 4096),
            nn.ReLU(inplace=True)
        )

        self.proj_layer2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )

        self.encoder = nn.Sequential()
        self.encoder.add_module('enc-layer1', self.enc_layer1) # input -> 64
        self.encoder.add_module('enc-layer2', self.enc_layer2) # 64 -> 64
        self.encoder.add_module('enc-layer3', self.enc_layer3) # 64 -> 128
        self.encoder.add_module('enc-layer4', self.enc_layer4) # 128 -> 128
        self.encoder.add_module('enc-layer5', self.enc_layer5) # 128 -> 256
        self.encoder.add_module('enc-layer6', self.enc_layer6) # 256 -> 256
        self.encoder.add_module('enc-layer7', self.enc_layer7) # 256 -> 256

        self.encoder.add_module('enc-layer8', self.enc_layer8) # 256 -> 512
        self.encoder.add_module('enc-layer9', self.enc_layer9) # 512 -> 512
        self.encoder.add_module('enc-layer10', self.enc_layer10) # 512 -> 512
        self.encoder.add_module('enc-layer11', self.enc_layer11) # 512 -> 512
        self.encoder.add_module('enc-layer12', self.enc_layer12) # 512 -> 512

        # self.projection_head = nn.Sequential()
        # self.projection_head.add_module('proj-layer1', self.proj_layer1)
        # self.projection_head.add_module('proj-layer2', self.proj_layer2)

    def forward(self, x):
        out = self.encoder(x)
        # print(f'Encoder output: {out.shape}')
        # out = torch.reshape(out, shape=(out.shape[0], -1))
        # print(f'Projection head input: {out.shape}')

        # out = self.projection_head(out)
        return out

class Classifier(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(input_channels, input_channels // 2),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(input_channels // 2, input_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.layer3 = nn.Linear(input_channels // 4, output_channels)

        self.fc = nn.Sequential()
        self.fc.add_module('layer1', self.layer1)
        self.fc.add_module('layer2', self.layer2)
        self.fc.add_module('layer3', self.layer3)

        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out
    
class ClassificationModel(nn.Module):
    def __init__(self,encoder_type,in_channels,out_channels,attention_mode= True):
        super().__init__()
        self.attention_mode = attention_mode
        if(encoder_type == 'resnet'):
            self.encoder = ResNet3D(in_channels,attention_mode=attention_mode)
            self.classifier = Classifier(32768,out_channels) 

        elif(encoder_type == 'vgg'):
            self.encoder  = VGG3D(in_channels)   
            self.classifier = Classifier(2048,out_channels) 

    def forward(self,x): 
        if(self.attention_mode):
            out,out_conv = self.encoder(x)
            out = torch.flatten(out,1,-1)
            out = self.classifier(out)
            return out,out_conv
        else:
            out = self.encoder(x)
            out = torch.flatten(out,1,-1)
            out = self.classifier(out)
            return out