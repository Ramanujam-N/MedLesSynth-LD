import torch
import torch.nn as nn
from ModelArchitecture.UNet import VGGBlock,ResConvBlock
from ModelArchitecture.Encoders import ResNet3DBasicBlock
class AutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_init_features = 64,block_type='resnet'):
        super().__init__()
        self.num_layers = num_layers
        self.num_init_features = num_init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.initial_layer = nn.Conv3d(in_channels,num_init_features,kernel_size=3,padding='same')
        self.final_layer = nn.Conv3d(num_init_features,out_channels,kernel_size=3,padding='same')
        
        # Encoder
        self.encoder1 = ResConvBlock(num_init_features,num_init_features*2)
        self.pool1 = nn.MaxPool3d(kernel_size = 2,stride = 2)
        self.encoder2 = ResConvBlock(num_init_features*2,num_init_features*4)
        self.pool2 = nn.MaxPool3d(kernel_size =2,stride = 2)
        self.encoder3 = ResConvBlock(num_init_features*4,num_init_features*8)
        self.pool3 = nn.MaxPool3d(kernel_size =2,stride = 2)

        # Bridge
        self.bottleneck = ResConvBlock(num_init_features*8, num_init_features*8)

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(num_init_features*8, num_init_features*4, kernel_size=2, stride=2)
        self.decoder3 = ResConvBlock(num_init_features*4,num_init_features*4)
        self.upconv2 = nn.ConvTranspose3d(num_init_features*4, num_init_features*2, kernel_size=2, stride=2)
        self.decoder2 = ResConvBlock(num_init_features*2,num_init_features*2)
        self.upconv1 = nn.ConvTranspose3d(num_init_features*1, num_init_features*1, kernel_size=2, stride=2)
        self.decoder1 = ResConvBlock(num_init_features*1,num_init_features*1)
        
        # Output
        self.outputconv = nn.Conv3d(num_init_features*1,out_channels,1,padding='same')
        self.outputnorm = nn.BatchNorm3d(out_channels)
        self.outputact = nn.Sigmoid()

    def forward(self,x):
        out_list = []
        for i in range(self.num_layers):
            out_list.append(self.encoder(x))
        out = self.decoder(out)
        return out

        
class VariationalEncoder3D(nn.Module):
    def __init__(self, image_channels,out_channels=1,latent_dims=1024):
        super().__init__()
        self.expansion = 1
        self.in_channels = 64

        self.conv1 = nn.Conv3d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # self.layer5 = self._make_layer(512, 1024, 2, stride=2)
        # self.layer6 = self._make_layer(1024, 2048, 2, stride=2)        
    
        # Calculation of KL Loss

        self.latent_dim = latent_dims
        self.enc_flat = 32768
        self.mu = nn.Linear(self.enc_flat,latent_dims)
        self.log_var = nn.Linear(self.enc_flat,latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda(0)
        self.N.scale = self.N.scale.cuda(0)
        self.kl =0
        self.decoder_lin = nn.Linear(latent_dims,self.enc_flat)
        self.bottlerelu = nn.ReLU()

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = ResNet3DBasicBlock(256,256)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = ResConvBlock(128,128)
        self.upconv2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ResConvBlock(64,64)
        self.upconv1 = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.decoder1 = ResNet3DBasicBlock(64,64)

        # Output
        self.outputupconv = nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2)
        self.outputconv = nn.Conv3d(64,out_channels,1,padding='same')
        self.outputnorm = nn.BatchNorm3d(out_channels)
        self.outputact = nn.Sigmoid()
        
        self.apply(self._init_weights)

    def _make_layer(self,in_channels, output_channels, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, output_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(output_channels),
            )
            
        
        layers = []
        layers.append(
            ResNet3DBasicBlock(
                in_channels,
                output_channels, 
                stride,
                self.expansion,
                downsample
            )
        )
        #self.in_channels = output_channels * self.expansion

        for _ in range(1, blocks):
            layers.append(
                ResNet3DBasicBlock(
                    output_channels,
                    output_channels,
                    expansion=self.expansion
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Bringing channels to 64
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print(x.shape)

        # Encoder Layers
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        # Bottleneck
        out = torch.flatten(x_4,1,-1)
        out = self.bottlerelu(out)

        mean = self.mu(out)
        log_var = self.log_var(out)
        sigma = torch.exp(log_var)

        eps =1e-7
        sample = mean + sigma*(self.N.sample(mean.shape)+eps)
        latent = self.decoder_lin(sample).reshape(x_4.shape)

        self.kl =  -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
        #self.kl = torch.sum((sigma**2 + mean**2 - log_var - 1/2),dim=-1).mean()

        # Decoder Layers
        out = self.upconv4(latent)
        out = self.decoder4(out)
        out = self.upconv3(out)
        out = self.decoder3(out)
        out = self.upconv2(out)
        out = self.decoder2(out)
        out = self.upconv1(out)
        out = self.decoder1(out)

        
        # output layer 
        out = self.outputupconv(out)
        out = self.outputconv(out)
        out = self.outputnorm(out)

        return out,(x_1,x_2,x_3,x_4)

    def returnmusigma(self):
        return self.mu,self.log_var
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            torch.nn.init.uniform_(module.weight,-0.08,0.08)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


# class VariationalEncoder(nn.Module):
#     def __init__(self, in_channels=1, latent_dims=1024, init_features=64):
#         super(VariationalEncoder, self).__init__()
#         features = init_features
#         self.encoder1 = VariationalEncoder._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder2 = VariationalEncoder._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder3 = VariationalEncoder._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
#         self.encoder4 = VariationalEncoder._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)

#         self.bottleneck = VariationalEncoder._block(features * 8, features * 16, name="bottleneck")

#         self.latent_dim = latent_dims
#         self.enc_flat = 65536
#         self.mu = nn.Linear(self.enc_flat,latent_dims)
#         self.log_var = nn.Linear(self.enc_flat,latent_dims)
#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda(0)
#         self.N.scale = self.N.scale.cuda(0)
#         self.kl =0

#         self.decoder_lin = nn.Linear(latent_dims,self.enc_flat)


#         self.upconv4 = nn.ConvTranspose3d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = VariationalEncoder._block((features * 8), features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose3d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = VariationalEncoder._block((features * 4) , features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose3d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = VariationalEncoder._block((features * 2) , features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose3d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = VariationalEncoder._block(features, features, name="dec1")

#         self.seg = nn.Conv3d( in_channels=features, out_channels=1, kernel_size=1)
#         self.outsig = nn.Sigmoid()

#     def forward(self, x):
#         enc1 = self.encoder1(x)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))


#         bottleneck = self.bottleneck(self.pool4(enc4))

#         #print(bottleneck.shape)
#         out = torch.flatten(bottleneck,1,-1)
#         #print(out.shape)
#         mean = self.mu(out)
#         log_var = self.log_var(out)
#         sigma = torch.exp(log_var)

#         eps =1e-6
#         sample = mean + sigma*(self.N.sample(mean.shape)+eps)
#         latent = self.decoder_lin(sample).reshape(bottleneck.shape)

#         self.kl = torch.sum((sigma**2 + mean**2 - log_var - 1/2),dim=-1).mean()


#         dec4 = self.upconv4(latent)
#         #dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         #dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         #dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         #dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         seg = self.seg(dec1)
#         return self.outsig(seg)

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv3d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=True,
#                         ),
#                     ),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (name + "norm1", nn.BatchNorm3d(num_features=features)),

#                     (
#                         name + "conv2",
#                         nn.Conv3d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=True,
#                         ),
#                     ),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                     (name + "norm2", nn.BatchNorm3d(num_features=features)),
#                 ]
#             )
#         )
#     def returnmusigma(self):
#         return self.mu,self.log_var