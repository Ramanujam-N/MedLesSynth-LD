import torch
import torch.nn as nn

# import tensorflow as tf
# from keras.layers import Conv2D, UpSampling2D
# from keras.layers import add
# from keras.models import Model
from ModelArchitecture.UNet2D import SpatialAttention, ChannelAttention
from CustomLayers.ConvBlock2D import Gen_Conv2d_Block,Conv2dSame

kernel_initializer = 'he_uniform'
interpolation = "nearest"

class DuckNet(nn.Module):
    def __init__(self,input_channels, out_classes, starting_filters):
        super().__init__()
        
        self.out_classes = out_classes
        self.starting_filters = starting_filters

        self.p1 = Conv2dSame(input_channels, starting_filters * 2, 2, stride=2)
        self.p2 = Conv2dSame(starting_filters * 2, starting_filters * 4, 2, stride=2)
        self.p3 = Conv2dSame(starting_filters * 4, starting_filters * 8, 2, stride=2)
        self.p4 = Conv2dSame(starting_filters * 8, starting_filters * 16, 2, stride=2)
        self.p5 = Conv2dSame(starting_filters * 16, starting_filters * 32, 2, stride=2)

#########################Attention before skip ####################
        # self.spa_t0 = SpatialAttention()
        # self.spa_t1 = SpatialAttention()
        # self.spa_t2 = SpatialAttention()
        # self.spa_t3 = SpatialAttention()
        # self.spa_t4 = SpatialAttention()

        # self.ch_t0 = ChannelAttention(starting_filters)
        # self.ch_t1 = ChannelAttention(starting_filters * 2)
        # self.ch_t2 = ChannelAttention(starting_filters * 4)
        # self.ch_t3 = ChannelAttention(starting_filters * 8)
        # self.ch_t4 = ChannelAttention(starting_filters * 16)
#############################################
################# Attention before duck block ##################
        # self.spa_p0 = SpatialAttention()
        # self.spa_p1 = SpatialAttention()
        # self.spa_p2 = SpatialAttention()
        # self.spa_p3 = SpatialAttention()
        # self.spa_p4 = SpatialAttention()

        # self.ch_p0 = ChannelAttention(starting_filters)
        # self.ch_p1 = ChannelAttention(starting_filters * 2)
        # self.ch_p2 = ChannelAttention(starting_filters * 4)
        # self.ch_p3 = ChannelAttention(starting_filters * 8)
        # self.ch_p4 = ChannelAttention(starting_filters * 16)
##############################################

        self.t0 = Gen_Conv2d_Block(input_channels,starting_filters, block_type='duck', repeat=1)

        self.l1i = Conv2dSame(starting_filters,starting_filters * 2, 2, stride=2)
        #self.s1 = add([l1i, p1])
        self.t1 = Gen_Conv2d_Block(starting_filters * 2,starting_filters * 2, block_type='duck', repeat=1)

        self.l2i = Conv2dSame(starting_filters * 2, starting_filters * 4, 2, stride=2)
        #self.s2 = add([l2i, p2])
        self.t2 = Gen_Conv2d_Block(starting_filters * 4,starting_filters * 4, block_type='duck', repeat=1)

        self.l3i = Conv2dSame(starting_filters * 4, starting_filters * 8, 2, stride=2)
        #s3 = add([l3i, p3])
        self.t3 = Gen_Conv2d_Block(starting_filters * 8,starting_filters * 8, block_type='duck', repeat=1)

        self.l4i = Conv2dSame(starting_filters * 8, starting_filters * 16, 2, stride=2)
        #s4 = add([l4i, p4])
        self.t4 = Gen_Conv2d_Block(starting_filters * 16,starting_filters * 16, block_type='duck', repeat=1)

        self.l5i = Conv2dSame(starting_filters * 16, starting_filters * 32, 2, stride=2)
        #s5 = add([l5i, p5])
        self.t51 = Gen_Conv2d_Block(starting_filters * 32,starting_filters * 32, block_type='resnet', repeat=2)
        self.t53 = Gen_Conv2d_Block(starting_filters * 32,starting_filters * 16, block_type='resnet', repeat=2)

        self.l5o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c4 = add([l5o, t4])
        self.q4 = Gen_Conv2d_Block(starting_filters * 16,starting_filters * 8, block_type='duck', repeat=1)

        self.l4o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c3 = add([l4o, t3])
        self.q3 = Gen_Conv2d_Block(starting_filters * 8,starting_filters * 4, block_type='duck', repeat=1)

        self.l3o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c2 = add([l3o, t2])
        self.q6 = Gen_Conv2d_Block(starting_filters * 4,starting_filters * 2, block_type='duck', repeat=1)

        self.l2o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c1 = add([l2o, t1])
        self.q1 = Gen_Conv2d_Block(starting_filters * 2,starting_filters, block_type='duck', repeat=1)

        self.l1o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c0 = add([l1o, t0])
        self.z1 = Gen_Conv2d_Block(starting_filters,starting_filters, block_type='duck', repeat=1)
        
        self.output = Conv2dSame(starting_filters,out_classes, (1, 1))
        self.acti = nn.Sigmoid()
        
        

    def forward(self,x):
        out_p1 = self.p1(x)
        out_p2 = self.p2(out_p1)
        out_p3 = self.p3(out_p2)
        out_p4 = self.p4(out_p3)
        out_p5 = self.p5(out_p4)

        #######################
        # out_p0 = self.ch_p0(self.spa_p0(out_p0))
        # out_p1 = self.ch_p1(self.spa_p1(out_p1))
        # out_p2 = self.ch_p2(self.spa_p2(out_p2))
        # out_p3 = self.ch_p3(self.spa_p3(out_p3))
        # out_p4 = self.ch_p4(self.spa_p4(out_p4))

        ########################
        out_t0 = self.t0(x)

        out_l1i = self.l1i(out_t0)
        s1 = out_l1i + out_p1
        out_t1 = self.t1(s1)

        out_l2i = self.l2i(out_t1)
        s2 = out_l2i + out_p2
        out_t2 = self.t2(s2)

        out_l3i = self.l3i(out_t2)
        s3 = out_l3i + out_p3
        out_t3 = self.t3(s3)

        out_l4i = self.l4i(out_t3)
        s4 = out_l4i + out_p4
        out_t4 = self.t4(s4)

        out_l5i = self.l5i(out_t4)
        s5 = out_l5i + out_p5
        out_t51 = self.t51(s5)
        out_t53 = self.t53(out_t51)


# ####################################

# Channel followed by spatial
        # out_t0 = self.spa_t0(self.ch_t0(out_t0))
        # out_t1 = self.spa_t1(self.ch_t1(out_t1))
        # out_t2 = self.spa_t2(self.ch_t2(out_t2))
        # out_t3 = self.spa_t3(self.ch_t3(out_t3))
        # out_t4 = self.spa_t4(self.ch_t4(out_t4))

#Spatial followed by channel

        # out_t0 = self.ch_t0(self.spa_t0(out_t0))
        # out_t1 = self.ch_t1(self.spa_t1(out_t1))
        # out_t2 = self.ch_t2(self.spa_t2(out_t2))
        # out_t3 = self.ch_t3(self.spa_t3(out_t3))
        # out_t4 = self.ch_t4(self.spa_t4(out_t4))

        # out_t0 = self.spa_t0((out_t0))
        # out_t1 = self.spa_t1((out_t1))
        # out_t2 = self.spa_t2((out_t2))
        # out_t3 = self.spa_t3((out_t3))
        # out_t4 = self.spa_t4((out_t4))


######################################



        out_l5o = self.l5o(out_t53)
        c4 = out_l5o + out_t4
        out_q4 = self.q4(c4)

        out_l4o = self.l4o(out_q4)
        c3 = out_l4o + out_t3
        out_q3 = self.q3(c3)
    
        out_l3o = self.l3o(out_q3)
        c2 = out_l3o + out_t2
        out_q6 = self.q6(c2)

        out_l2o = self.l2o(out_q6)
        c1 = out_l2o + out_t1
        out_q1 = self.q1(c1)

        out_l1o = self.l1o(out_q1)
        c0 = out_l1o + out_t0
        out_z1 = self.z1(c0)
        
        output = self.output(out_z1)
        output = self.acti(output)

        return output
    
import time
class DuckNet_smaller(nn.Module):
    def __init__(self,input_channels, out_classes, starting_filters):
        super().__init__()
        
        self.out_classes = out_classes
        self.starting_filters = starting_filters

        self.p1 = Conv2dSame(input_channels, starting_filters * 2, 2, stride=2)
        self.p2 = Conv2dSame(starting_filters * 2, starting_filters * 4, 2, stride=2)
        self.p3 = Conv2dSame(starting_filters * 4, starting_filters * 8, 2, stride=2)
        self.p4 = Conv2dSame(starting_filters * 8, starting_filters * 16, 2, stride=2)


        self.t0 = Gen_Conv2d_Block(input_channels,starting_filters, block_type='duck', repeat=1)

        self.l1i = Conv2dSame(starting_filters,starting_filters * 2, 2, stride=2)
        #self.s1 = add([l1i, p1])
        self.t1 = Gen_Conv2d_Block(starting_filters * 2,starting_filters * 2, block_type='duck', repeat=1)

        self.l2i = Conv2dSame(starting_filters * 2, starting_filters * 4, 2, stride=2)
        #self.s2 = add([l2i, p2])
        self.t2 = Gen_Conv2d_Block(starting_filters * 4,starting_filters * 4, block_type='duck', repeat=1)

        self.l3i = Conv2dSame(starting_filters * 4, starting_filters * 8, 2, stride=2)
        #s3 = add([l3i, p3])
        self.t31 = Gen_Conv2d_Block(starting_filters * 8,starting_filters * 8, block_type='resnet', repeat=2)
        
        self.t33 = Gen_Conv2d_Block(starting_filters * 8,starting_filters * 4, block_type='resnet', repeat=2)

        self.l3o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c2 = add([l3o, t2])
        self.q6 = Gen_Conv2d_Block(starting_filters * 4,starting_filters * 2, block_type='duck', repeat=1)

        self.l2o = nn.Upsample(scale_factor = (2, 2), mode=interpolation)
        #c1 = add([l2o, t1])
        self.q1 = Gen_Conv2d_Block(starting_filters * 2,starting_filters, block_type='duck', repeat=1)

        self.l1o = nn.Upsample(scale_factor = (2,2), mode=interpolation)
        #c0 = add([l1o, t0])
        self.z1 = Gen_Conv2d_Block(starting_filters,starting_filters, block_type='duck', repeat=1)
        
        self.output = Conv2dSame(starting_filters,out_classes, (1, 1))
        self.acti = nn.Sigmoid()
        
        

    def forward(self,x):

        out_p1 = self.p1(x)
        out_p2 = self.p2(out_p1)
        out_p3 = self.p3(out_p2)

        out_t0 = self.t0(x)

        out_l1i = self.l1i(out_t0)
        s1 = out_l1i + out_p1
        out_t1 = self.t1(s1)

        out_l2i = self.l2i(out_t1)
        s2 = out_l2i + out_p2
        out_t2 = self.t2(s2)

        out_l3i = self.l3i(out_t2)
        s3 = out_l3i + out_p3
        
        out_t31 = self.t31(s3)
        out_t33 = self.t33(out_t31)


        out_l3o = self.l3o(out_t33)
        c2 = out_l3o + out_t2
        out_q6 = self.q6(c2)

        out_l2o = self.l2o(out_q6)
        c1 = out_l2o + out_t1
        out_q1 = self.q1(c1)

        out_l1o = self.l1o(out_q1)
        c0 = out_l1o + out_t0
        out_z1 = self.z1(c0)
        
        output = self.output(out_z1)
        output = self.acti(output)

        return output


def create_model(img_height,img_width,img_depth,input_channels,out_classes,starting_filters):
    
    def weight_init(m):
        if isinstance(m, Conv2dSame) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(m.bias)

    ducknet = DuckNet()
    ducknet.apply(weight_init)

    return ducknet


# def create_model(img_height, img_width, input_chanels, out_classes, starting_filters):
#     input_layer = tf.keras.layers.Input((img_height, img_width, input_chanels))

#     print('Starting DUCK-Net')

#     p1 = Conv2D(starting_filters * 2, 2, stride=2, padding='same')(input_layer)
#     p2 = Conv2D(starting_filters * 4, 2, stride=2, padding='same')(p1)
#     p3 = Conv2D(starting_filters * 8, 2, stride=2, padding='same')(p2)
#     p4 = Conv2D(starting_filters * 16, 2, stride=2, padding='same')(p3)
#     p5 = Conv2D(starting_filters * 32, 2, stride=2, padding='same')(p4)

#     t0 = conv_block_2D(input_layer, starting_filters, 'duckv2', repeat=1)

#     l1i = Conv2D(starting_filters * 2, 2, stride=2, padding='same')(t0)
#     s1 = add([l1i, p1])
#     t1 = conv_block_2D(s1, starting_filters * 2, 'duckv2', repeat=1)

#     l2i = Conv2D(starting_filters * 4, 2, stride=2, padding='same')(t1)
#     s2 = add([l2i, p2])
#     t2 = conv_block_2D(s2, starting_filters * 4, 'duckv2', repeat=1)

#     l3i = Conv2D(starting_filters * 8, 2, stride=2, padding='same')(t2)
#     s3 = add([l3i, p3])
#     t3 = conv_block_2D(s3, starting_filters * 8, 'duckv2', repeat=1)

#     l4i = Conv2D(starting_filters * 16, 2, stride=2, padding='same')(t3)
#     s4 = add([l4i, p4])
#     t4 = conv_block_2D(s4, starting_filters * 16, 'duckv2', repeat=1)

#     l5i = Conv2D(starting_filters * 32, 2, stride=2, padding='same')(t4)
#     s5 = add([l5i, p5])
#     t51 = conv_block_2D(s5, starting_filters * 32, 'resnet', repeat=2)
#     t53 = conv_block_2D(t51, starting_filters * 16, 'resnet', repeat=2)

#     l5o = UpSampling2D((2, 2), interpolation=interpolation)(t53)
#     c4 = add([l5o, t4])
#     q4 = conv_block_2D(c4, starting_filters * 8, 'duckv2', repeat=1)

#     l4o = UpSampling2D((
#     l1o = UpSampling2D((2, 2), interpolation=interpolation)(q1)
#     c0 = add([l1o, t0])
#     z1 = conv_block_2D(c0, starting_filters, 'duckv2', repeat=1)

#     output = Conv2D(out_classes, (1, 1), activation='sigmoid')(z1)

#     model = Model(inputs=input_layer, outputs=output)

#     return model2, 2), interpolation=interpolation)(q4)
#     c3 = add([l4o, t3])
#     q3 = conv_block_2D(c3, starting_filters * 4, 'duckv2', repeat=1)

#     l3o = UpSampling2D((2, 2), interpolation=interpolation)(q3)
#     c2 = add([l3o, t2])
#     q6 = conv_block_2D(c2, starting_filters * 2, 'duckv2', repeat=1)

#     l2o = UpSampling2D((2, 2), interpolation=interpolation)(q6)
#     c1 = add([l2o, t1])
#     q1 = conv_block_2D(c1, starting_filters, 'duckv2', repeat=1)
