#####################################################################################################################################
# This file contains the network architecture of the proposed 3D CNN hybrid network. The network is designed to accept volume chunks.
#####################################################################################################################################

from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation,Dense,Input,BatchNormalization,Conv3D, Flatten,\
AveragePooling3D,MaxPooling3D,Dropout,Reshape,Lambda,GlobalAveragePooling3D
#from keras.layers import Dense
#from keras.layers import Input
#from keras.layers import BatchNormalization
#from keras.layers import Conv3D, Flatten
#from keras.layers import MaxPooling3D
#from keras.layers import AveragePooling3D
#from keras.layers import Dropout
#from keras.layers import Reshape
#from keras.layers import Lambda
#from keras.layers import GlobalAveragePooling3D
from tensorflow.keras.regularizers import l2
from keras.engine.topology import get_source_inputs
#from keras.utils import layer_utils
#from keras.utils.data_utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
#from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D

def conv3d_bn(x,
              filters,
              strides=(1, 1, 1),
              padding='same'
              ):
    """Utility function to apply conv3d + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """

    x = Conv3D(filters, (3, 3, 3),
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    return x

def conv3d_bn_1x1(x,
              filters,
              strides=(1, 1, 1),
              padding='same'
              ):
    """Utility function to apply conv3d + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv3D`.
        num_frames: frames (time depth) of the convolution kernel.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv3D`.
        strides: strides in `Conv3D`.
        use_bias: use bias or not  
        use_activation_fn: use an activation function or not.
        use_bn: use batch normalization or not.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv3D` and `BatchNormalization`.
    """

    x = Conv3D(filters, (1, 1, 1),
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    return x


def _bn_relu(input):
    CHANNEL_AXIS = -1
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def fire_module(x, squeeze=16, expand=64):

    channel_axis = -1

    x = Conv3D(squeeze, (1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = Activation('relu')(x)

    left = Conv3D(expand, (1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    left = Activation('relu')(left)

    right = Conv3D(expand, (3, 3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)

    right = Activation('relu')(right)

    x = layers.concatenate([left, right], axis=channel_axis)
    return x

def Inception_Inflated3d(img_input):
    channel_axis = -1
    #Instantiates the Inflated 3D Inception v1 architecture.

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, strides=(2, 2, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn_1x1(x, 64,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 96, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 128, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn_1x1(x, 16,strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 32,strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 32, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')
       
    # Mixed 3c
    branch_0 = conv3d_bn_1x1(x, 128, strides=(1, 1, 1),padding='same')

    branch_1 = conv3d_bn_1x1(x, 128,strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 192, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn_1x1(x, 32, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 96, strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')

    # Downsampling (spatial and temporal)
   

    x = fire_module(x, squeeze=16, expand=64)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 1), padding='same')(x)
    # Mixed 4b
    branch_0 = conv3d_bn_1x1(x, 192, strides=(1, 1, 1),padding='same')

    branch_1 = conv3d_bn_1x1(x, 96, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 208,strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn_1x1(x, 16, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 48,strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')


    # Mixed 4c
    branch_0 = conv3d_bn_1x1(x, 160, strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 112, strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 224, strides=(1, 1, 1), padding='same')

    branch_2 = conv3d_bn_1x1(x, 24, strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 64, strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 64, strides=(1, 1, 1), padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')

    x = fire_module(x, squeeze=16, expand=64)
    # Mixed 4d
    branch_0 = conv3d_bn_1x1(x, 128,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 128, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 256, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn_1x1(x, 24, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 64, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')

    # Mixed 4e
    branch_0 = conv3d_bn_1x1(x, 112,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 144, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 288, padding='same')

    branch_2 = conv3d_bn_1x1(x, 32,strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 64, padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4e')

    x = fire_module(x, squeeze=32, expand=128)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    # Mixed 4f
    branch_0 = conv3d_bn_1x1(x, 256,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 160, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 320, padding='same')

    branch_2 = conv3d_bn_1x1(x, 32,strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 128, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 128, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')

    # Downsampling (spatial and temporal)

    

    
    # Mixed 5b
    branch_0 = conv3d_bn_1x1(x, 256,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn_1x1(x, 160,strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 320, padding='same')

    branch_2 = conv3d_bn_1x1(x, 32, strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 128, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn_1x1(branch_3, 128, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')
    x = fire_module(x, squeeze=64, expand=256)
    x = GlobalAveragePooling3D()(x)

    return x

######################################################################
# Input: patchSize (image size of the volume chunk)
# Output: 3D hzbrid CNN model
######################################################################

def createModel(patchSize):
    print('entering create model this version of the network  uses 1x1 conv layers ')
    input_tensor = Input(shape=(patchSize[0], patchSize[1],patchSize[2], 1))


    
    output = Inception_Inflated3d(img_input=input_tensor)
    input_gender = Input(shape=(1,), name='gender_input')

    x = layers.concatenate([output, input_gender])
   # input_gender = Input(shape=(1,), name='gender_input')

    #x = layers.concatenate([output, input_gender])
    

#    x = Dropout(0.5)(output)
#    x = Dense(1024)(x)
    #x = Dropout(0.1)(output)
    x = Dense(512)(x)
    #x = Dropout(0.1)(x)
    x = Dense(256)(x)
#    x = Dropout(0.5)(x)
    x = Dense(128)(x)


    # fully-connected layer
    output = Dense(units=1,
                   activation='linear',
                   name='regression')(x)
#    output = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
#                  kernel_initializer='he_normal', activation='softmax')(x)
    sModelName = 'BioAgeNet_Regression'
    cnn = Model([input_tensor, input_gender ], output, name=sModelName)
    return cnn, sModelName

if __name__ == '__main__':
    model,_ = createModel([121,145, 12, 1])
    plot_model(model, to_file='F:/Final_Codes/Biological_Age_Estimation_Brain/network/3D_hybrid_CNN_network_gender.png', show_shapes=True)
    model.summary()
