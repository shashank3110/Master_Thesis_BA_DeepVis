'''
Pytorch implementation of Brain CA Estiimation.

'''
from __future__ import print_function
from __future__ import absolute_import

import warnings
import os
# import utils.gen_3D_vol_slices_age_gender_torch as generator

import math
import numpy as np
import torch
import torchvision
import torch.utils.data
from torch.utils.data import DataLoader,TensorDataset
import logging
from keras.layers import Input
from keras.layers.merge import concatenate
import torch.nn as nn
from torch.nn import BatchNorm3d,Conv3d,ReLU,MaxPool3d,Linear
import torch.nn.functional as F
# fg = features,genders

# fg,labels=generator.batch_and_run(train_generator, batch_size, count_train, case='train')
# train_data = TensorDataset(fg,labels)
# train_data = DataLoader(train_data,batch_size)
def compute_pad(stride=(1,1,1),kernel_size=(3,3,3),dim=0, s=0):
        if s % stride[dim] == 0:
            return max(kernel_size[dim] - stride[dim], 0)
        else:
            return max(kernel_size[dim] - (s % stride[dim]), 0)
def pad_same(x):
        # compute 'same' padding
        stride=(1,1,1)
        kernel_size=(3,3,3)
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(stride[0]))
        out_h = np.ceil(float(h) / float(stride[1]))
        out_w = np.ceil(float(w) / float(stride[2]))
        #print out_t, out_h, out_w
        pad_t = compute_pad(stride,kernel_size,0, t)
        pad_h = compute_pad(stride,kernel_size,1, h)
        pad_w = compute_pad(stride,kernel_size,2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return x


def Conv3D_BN(x,in_channels,
              filters,
              strides=(1, 1, 1),
              padding='same'
              ):
    """Utility function to apply conv3d + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        in_channels : eg: 3 for RGB image check image dimensions before passing
        this value
        num_frames: frames (time depth) of the convolution kernel.
        
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """
    x=pad_same(x)
    x = Conv3d(in_channels=in_channels,out_channels=filters, kernel_size=(3, 3, 3),
        stride=strides,
        
        bias=False)(x)

    # bn_axis = -1
    #verify this
    # One problem is we cannot specify on which axis to perform
    #batch norm in pytorch it performs on all axes
    x = BatchNorm3d(num_features=filters)(x)
    x = ReLU()(x)

    return x
def MaxPool3D_SamePadding(x,kernel_size=(3, 3, 3), stride=(1, 1, 1), padding='same',name=''):

    x=pad_same(x)
    x=MaxPool3d(kernel_size,stride)(x)
    return x
####################  

class Conv3D_BN_Same_Padding(Conv3d):
    
    def __init__(self,in_channels=1,out_channels=64, kernel_size=(3, 3, 3),
        stride=(1,1,1),name=''):
        super(Conv3D_BN_Same_Padding,self).__init__(in_channels,out_channels, kernel_size,
        stride,)
        self.name = name
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels= out_channels
        self.stride =  stride
        self.BatchNorm3d =BatchNorm3d(num_features=self.out_channels)
        self.ReLU = ReLU()

    def forward(self, x):
        x=pad_same(x)
        x = Conv3d(in_channels=self.in_channels,out_channels=self.out_channels, kernel_size=self.kernel_size,
        stride=self.stride,
        
        bias=False)(x)
        x = self.BatchNorm3d(x)
        x = self.ReLU(x)

        return x

class MaxPool3D_Same_Padding(MaxPool3d):
    
    def __init__(self,kernel_size=(3,3,3),padding=0,stride=(1,1,1),name=''):
        super(MaxPool3D_Same_Padding,self).__init__(kernel_size,padding,stride)
        self.name = name
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride =  stride
    

    def forward(self, x):
        x=pad_same(x)
        x=MaxPool3d(kernel_size=self.kernel_size,stride=self.stride)(x)
        return x
###################

#############################


def regularize_kernel(l=5e-4,w=None):
    return w+ l*torch.sum(w**2)
    
class Fire_Module(nn.Module):
    def __init__(self,in_channels=1,squeeze=16,expand=64,name=''):

        super(Fire_Module,self).__init__()
        self.in_channels=in_channels
        self.squeeze=squeeze #16
        self.expand=expand #64
        self.Conv3D_squeeze = Conv3d(self.in_channels,self.squeeze, kernel_size=(1, 1, 1), padding=0, bias=False)
        
        self.in_channels = self.squeeze
        self.Conv3D_expand_left = Conv3d(self.in_channels,self.expand, kernel_size=(1, 1, 1), padding=0, bias=False)
        
    
        self.Conv3D_expand_right = Conv3d(self.in_channels,self.expand,kernel_size= (3, 3, 3), padding=0, bias=False)
        

        self.activation = ReLU()
        self.name=name

    def forward(self,x):
        
        #print(self.Conv3D_squeeze.weight)
        #initialize_kernel
        torch.nn.init.kaiming_normal_(self.Conv3D_squeeze.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #regularize_kernel
        self.Conv3D_squeeze.weight.data=regularize_kernel(w=self.Conv3D_squeeze.weight)
        # print(self.Conv3D_squeeze.weight)
        x= self.Conv3D_squeeze(x)

        x=self.activation(x)



        #initialize_kernel
        torch.nn.init.kaiming_normal_(self.Conv3D_expand_left.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #regularize_kernel
        self.Conv3D_expand_left.weight.data=regularize_kernel(w=self.Conv3D_expand_left.weight)


        left= self.Conv3D_expand_left(x)
        left=self.activation(left)

        x=pad_same(x)

        #initialize_kernel
        torch.nn.init.kaiming_normal_(self.Conv3D_expand_right.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        #regularize_kernel
        self.Conv3D_expand_right.weight.data=regularize_kernel(w=self.Conv3D_expand_right.weight)
        right= self.Conv3D_expand_right(x)

        right=self.activation(right)
        
        x= torch.cat([left,right],dim=1)
        print(f'torch fire cat shape={x.shape}')
        return x

class Inception_Inflated3D_Augmented(nn.Module):

    def __init__(self):
        super(Inception_Inflated3D_Augmented,self).__init__()
        # self.img_input=img_input
        self.Relu = ReLU()
        
        #Base Conv
        self.Conv3D_BN1 = Conv3D_BN_Same_Padding(in_channels=1,out_channels=64,stride=(2,2,1))
        self.Conv3D_BN2 = Conv3D_BN_Same_Padding(in_channels=64,out_channels=64,stride=(1,1,1))
        self.Conv3D_BN3 = Conv3D_BN_Same_Padding(in_channels=64,out_channels=192,stride=(1,1,1))

        #Branch Conv

        #3b
        self.Conv3D_BN3b0 = Conv3D_BN_Same_Padding(in_channels=192,out_channels=96,stride=(1,1,1))#x
        self.Conv3D_BN3b1 = Conv3D_BN_Same_Padding(in_channels=192,out_channels=16,stride=(1,1,1))#x
        self.Conv3D_BN3b1_1 = Conv3D_BN_Same_Padding(in_channels=96,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN3b2 = Conv3D_BN_Same_Padding(in_channels=192,out_channels=64,stride=(1,1,1))#x
        self.Conv3D_BN3b2_1 = Conv3D_BN_Same_Padding(in_channels=16,out_channels=32,stride=(1,1,1))
        self.Conv3D_BN3b3 = Conv3D_BN_Same_Padding(in_channels=192,out_channels=32,stride=(1,1,1))#x
 

        #3c
        self.Conv3D_BN3c0 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN3c1 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=32,stride=(1,1,1))
        self.Conv3D_BN3c1_1 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=192,stride=(1,1,1))
        self.Conv3D_BN3c2 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN3c2_1 = Conv3D_BN_Same_Padding(in_channels=32,out_channels=96,stride=(1,1,1))
        self.Conv3D_BN3c3 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=64,stride=(1,1,1))


        #4b
        self.Conv3D_BN4b0 = Conv3D_BN_Same_Padding(in_channels=480,out_channels=96,stride=(1,1,1))
        self.Conv3D_BN4b1 = Conv3D_BN_Same_Padding(in_channels=480,out_channels=16,stride=(1,1,1))
        self.Conv3D_BN4b1_1 = Conv3D_BN_Same_Padding(in_channels=96,out_channels=208,stride=(1,1,1))
        self.Conv3D_BN4b2 = Conv3D_BN_Same_Padding(in_channels=480,out_channels=192,stride=(1,1,1))
        self.Conv3D_BN4b2_1 = Conv3D_BN_Same_Padding(in_channels=16,out_channels=48,stride=(1,1,1))
        self.Conv3D_BN4b3 = Conv3D_BN_Same_Padding(in_channels=480,out_channels=64,stride=(1,1,1))
        #fire1
        #4c
        self.Conv3D_BN4c0 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=112,stride=(1,1,1))
        self.Conv3D_BN4c1 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=24,stride=(1,1,1))
        self.Conv3D_BN4c1_1 = Conv3D_BN_Same_Padding(in_channels=112,out_channels=224,stride=(1,1,1))
        self.Conv3D_BN4c2 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=160,stride=(1,1,1))
        self.Conv3D_BN4c2_1 = Conv3D_BN_Same_Padding(in_channels=24,out_channels=64,stride=(1,1,1))
        self.Conv3D_BN4c3 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=64,stride=(1,1,1))

        #4d
        self.Conv3D_BN4d0 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN4d1 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=24,stride=(1,1,1))
        self.Conv3D_BN4d1_1 = Conv3D_BN_Same_Padding(in_channels=128,out_channels=256,stride=(1,1,1))
        self.Conv3D_BN4d2 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN4d2_1 = Conv3D_BN_Same_Padding(in_channels=24,out_channels=64,stride=(1,1,1))
        self.Conv3D_BN4d3 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=64,stride=(1,1,1))

        #4e
        self.Conv3D_BN4e0 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=144,stride=(1,1,1))
        self.Conv3D_BN4e1 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=32,stride=(1,1,1))
        self.Conv3D_BN4e1_1 = Conv3D_BN_Same_Padding(in_channels=144,out_channels=288,stride=(1,1,1))
        self.Conv3D_BN4e2 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=112,stride=(1,1,1))
        self.Conv3D_BN4e2_1 = Conv3D_BN_Same_Padding(in_channels=32,out_channels=64,stride=(1,1,1))
        self.Conv3D_BN4e3 = Conv3D_BN_Same_Padding(in_channels=512,out_channels=64,stride=(1,1,1))
        
        #4f
        self.Conv3D_BN4f0 = Conv3D_BN_Same_Padding(in_channels=528,out_channels=160,stride=(1,1,1))
        self.Conv3D_BN4f1 = Conv3D_BN_Same_Padding(in_channels=528,out_channels=32,stride=(1,1,1))
        self.Conv3D_BN4f1_1 = Conv3D_BN_Same_Padding(in_channels=160,out_channels=320,stride=(1,1,1))
        self.Conv3D_BN4f2 = Conv3D_BN_Same_Padding(in_channels=528,out_channels=256,stride=(1,1,1))
        self.Conv3D_BN4f2_1 = Conv3D_BN_Same_Padding(in_channels=32,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN4f3 = Conv3D_BN_Same_Padding(in_channels=528,out_channels=128,stride=(1,1,1))
        #fire2
        #5b
        self.Conv3D_BN5b0 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=160,stride=(1,1,1))
        self.Conv3D_BN5b1 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=32,stride=(1,1,1))
        self.Conv3D_BN5b1_1 = Conv3D_BN_Same_Padding(in_channels=160,out_channels=320,stride=(1,1,1))
        self.Conv3D_BN5b2 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=256,stride=(1,1,1))
        self.Conv3D_BN5b2_1 = Conv3D_BN_Same_Padding(in_channels=32,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN5b3 = Conv3D_BN_Same_Padding(in_channels=256,out_channels=128,stride=(1,1,1))
        

        #5c
        self.Conv3D_BN5c0 = Conv3D_BN_Same_Padding(in_channels=832,out_channels=192,stride=(1,1,1))
        self.Conv3D_BN5c1 = Conv3D_BN_Same_Padding(in_channels=832,out_channels=48,stride=(1,1,1))
        self.Conv3D_BN5c1_1 = Conv3D_BN_Same_Padding(in_channels=192,out_channels=384,stride=(1,1,1))
        self.Conv3D_BN5c2 = Conv3D_BN_Same_Padding(in_channels=832,out_channels=384,stride=(1,1,1))
        self.Conv3D_BN5c2_1 = Conv3D_BN_Same_Padding(in_channels=48,out_channels=128,stride=(1,1,1))
        self.Conv3D_BN5c3 = Conv3D_BN_Same_Padding(in_channels=832,out_channels=128,stride=(1,1,1))
        #fire3

        #Max Pool3D
        self.MaxPool3D_SamePadding1 = MaxPool3D_Same_Padding(kernel_size=(3, 3, 3), stride=(1, 1, 1))
        self.MaxPool3D_SamePadding2 = MaxPool3D_Same_Padding(kernel_size=(3, 3, 3),stride=(2, 2, 1))
        self.MaxPool3D_SamePadding3 = MaxPool3D_Same_Padding((3, 3, 3), stride=(2, 2, 2))
        self.MaxPool3D_SamePadding4 = MaxPool3D_Same_Padding((2, 2, 2), stride=(2, 2, 1))

        #Fire Modules
        self.Fire_Module1=Fire_Module(512, squeeze=16, expand=64)
        self.Fire_Module2=Fire_Module(832, squeeze=32, expand=128)
        self.Fire_Module3=Fire_Module(1024, squeeze=64, expand=256)

        # #Regression Layers
        self.Dense1 = Linear(513,512)
        self.Dense2 = Linear(512,256)
        self.Dense3 = Linear(256,128)
        self.Dense4 = Linear(128,1)
        
    # bn_axis = -1
    #verify this
    # One problem is we cannot specify on which axis to perform
    #batch norm in pytorch it performs on all axes
    def forward(self,X):
        x = X[0]
        gender_tensor = X[1]
        channel_axis = 1 #-1
        dim = x.size()


        # x = Conv3D_BN(pad_same(x),1, 64, strides=(2, 2, 1)) #'same')
        x = self.Conv3D_BN1(x)

        # Downsampling (spatial only)
     
        x = self.MaxPool3D_SamePadding2(x)
        print(f'1 shape{x.shape}')
        # x = Conv3D_BN(x, 64,64, strides=(1, 1, 1))
        x = self.Conv3D_BN2(x)
        print(f'2 shape{x.shape}')
        # x=pad_same(x)
        # x = Conv3D_BN(x,64, 192, strides=(1, 1, 1))
        x = self.Conv3D_BN3(x)

        # Downsampling (spatial only)
        x = self.MaxPool3D_SamePadding2(x)


        # Mixed 3b
        # x=pad_same(x)
        print(f'3 shape{x.shape}')
        # branch_0 = Conv3D_BN(x,192,64,strides=(1, 1, 1))
        branch_0 = self.Conv3D_BN3b0(x)
        # x=pad_same(x)
        # print(branch_0.shape)
        # branch_1 = Conv3D_BN(x,192, 96, strides=(1, 1, 1))
        branch_1 = self.Conv3D_BN3b1(x)

        # x=pad_same(x)
        # branch_1 = Conv3D_BN(branch_1,96, 128, strides=(1, 1, 1))
        branch_0 = self.Conv3D_BN3b1_1(branch_0)
        # x=pad_same(x)
        # branch_2 = Conv3D_BN(x,192, 16,strides=(1, 1, 1), )
        branch_2 = self.Conv3D_BN3b2(x)
        # x=pad_same(x)
        # branch_2 = Conv3D_BN(branch_2,16, 32,strides=(1, 1, 1))
        branch_1 = self.Conv3D_BN3b2_1(branch_1)


        
        branch_3 = self.MaxPool3D_SamePadding1(x)
        # print(branch_3.shape)
        # x=pad_same(x)
        # branch_3 = Conv3D_BN(branch_3,192, 32, strides=(1, 1, 1))
        branch_3 = self.Conv3D_BN3b3(branch_3)
        print(branch_0.shape,branch_1.shape,branch_2.shape,branch_3.shape)
        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
           )
        
        # Mixed 3c

        # x=pad_same(x)
        print(f'3b shape{x.shape}')
        # branch_0 = Conv3D_BN(x,256, 128, strides=(1, 1, 1))
        branch_0 = self.Conv3D_BN3c0(x)
        branch_1 = self.Conv3D_BN3c1(x)
        branch_0 = self.Conv3D_BN3c1_1(branch_0)
        branch_2 = self.Conv3D_BN3c2(x)
        branch_1 = self.Conv3D_BN3c2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN3c3(branch_3)
        
        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
            )
        print(f'3c shape{x.shape}')
        # Downsampling (spatial and temporal)
        x = self.MaxPool3D_SamePadding3(x)
        print(x.shape)
        # Mixed 4b

        branch_0 = self.Conv3D_BN4b0(x)
        branch_1 = self.Conv3D_BN4b1(x)
        branch_0 = self.Conv3D_BN4b1_1(branch_0)
        branch_2 = self.Conv3D_BN4b2(x)
        branch_1 = self.Conv3D_BN4b2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN4b3(branch_3)
        

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
           )
        print(f'4b  shape{x.shape}')
        # x = Fire_Module(512, squeeze=16, expand=64)(x)
        x = self.Fire_Module1(x)
        print(f'4b + fire shape{x.shape}')
        # Mixed 4c

        branch_0 = self.Conv3D_BN4c0(x)
        branch_1 = self.Conv3D_BN4c1(x)
        branch_0 = self.Conv3D_BN4c1_1(branch_0)
        branch_2 = self.Conv3D_BN4c2(x)
        branch_1 = self.Conv3D_BN4c2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN4c3(branch_3)
        

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
            )
        print(f'4c shape{x.shape}')
        # Mixed 4d
        branch_0 = self.Conv3D_BN4d0(x)
        branch_1 = self.Conv3D_BN4d1(x)
        branch_0 = self.Conv3D_BN4d1_1(branch_0)
        branch_2 = self.Conv3D_BN4d2(x)
        branch_1 = self.Conv3D_BN4d2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN4d3(branch_3)
        
        
        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
         )
        print(f'4d shape{x.shape}')
        # Mixed 4e

        branch_0 = self.Conv3D_BN4e0(x)
        branch_1 = self.Conv3D_BN4e1(x)
        branch_0 = self.Conv3D_BN4e1_1(branch_0)
        branch_2 = self.Conv3D_BN4e2(x)
        branch_1 = self.Conv3D_BN4e2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN4e3(branch_3)
        

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
            )
        print(branch_0.shape, branch_1.shape, branch_2.shape, branch_3.shape)
        print(f'4e shape{x.shape}')
        # Mixed 4f
        branch_0 = self.Conv3D_BN4f0(x)
        branch_1 = self.Conv3D_BN4f1(x)
        branch_0 = self.Conv3D_BN4f1_1(branch_0)
        branch_2 = self.Conv3D_BN4f2(x)
        branch_1 = self.Conv3D_BN4f2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN4f3(branch_3)
        

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
            )
        print(branch_0.shape, branch_1.shape, branch_2.shape, branch_3.shape)
        print(f'4f shape{x.shape}')
        # Downsampling (spatial and temporal)
        x = self.MaxPool3D_SamePadding2(x)
        print(f'4f +max pool shape{x.shape}')
        # x = Fire_Module(832, squeeze=32, expand=128)(x)
        x = self.Fire_Module2(x)
        print(f'4f +fire shape{x.shape}')
        # Mixed 5b
        branch_0 = self.Conv3D_BN5b0(x)
        branch_1 = self.Conv3D_BN5b1(x)
        branch_0 = self.Conv3D_BN5b1_1(branch_0)
        branch_2 = self.Conv3D_BN5b2(x)
        branch_1 = self.Conv3D_BN5b2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN5b3(branch_3)
        

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
            )
        print(branch_0.shape, branch_1.shape, branch_2.shape, branch_3.shape)
        print(f'5b shape{x.shape}')
        # Mixed 5c

        branch_0 = self.Conv3D_BN5c0(x)
        branch_1 = self.Conv3D_BN5c1(x)
        branch_0 = self.Conv3D_BN5c1_1(branch_0)
        branch_2 = self.Conv3D_BN5c2(x)
        branch_1 = self.Conv3D_BN5c2_1(branch_1)
        branch_3 = self.MaxPool3D_SamePadding1(x)
        branch_3 = self.Conv3D_BN5c3(branch_3)
  

        x = torch.cat(
            [branch_2, branch_0, branch_1, branch_3],
            dim=channel_axis,
     )
        print(branch_0.shape, branch_1.shape, branch_2.shape, branch_3.shape)
        print(f'5c shape{x.shape}')
        ##x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
        x = self.Fire_Module3(x)
        # x = Fire_Module(x.shape[1], squeeze=64, expand=256)(x)
        print(f'5c + fire module shape{x.shape}')
        # x = F.avg_pool3d(x,(dim[0],dim[1],1),stride=1)
        #############
        # both methods below seem correct for global average pooling 3D use any of these 
        # x = F.avg_pool3d(x,(x.shape[2:]),stride=1).view(x.size()[0],-1)
        x = F.adaptive_avg_pool3d(x,output_size=dim[1]).view(x.size()[0],-1)
        #############
        print(f'avg pool shape={x.shape}')
        ###
        # both methods below seem correct
        x=nn.Flatten()(x)
        # x=x.view(1,x.shape[1])
        ###
        print(f'inflated output={x.shape}')

        ####
        print(f'Entering Regression Layers')
        x = torch.cat([x,gender_tensor.float()],dim =1)
        print(f'input + gender concat shape={x.shape}')
        x = self.Dense1(x)
        print(f'Dense1 shape={x.shape}')
        x = self.Dense2(x)
        print(f'Dense2 shape={x.shape}')
        x = self.Dense3(x)
        print(f'Dense3 shape={x.shape}')
        x = self.Dense4(x)
        print(f'Dense4 shape={x.shape}')

        ####

        return x
