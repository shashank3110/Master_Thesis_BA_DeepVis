"""Inception-v1 Inflated 3D ConvNet used for Kinetics CVPR paper.

The model is introduced in:

Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
Joao Carreira, Andrew Zisserman
https://arxiv.org/abs/1705.07750v1
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

import numpy as np

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv3D, Flatten
from keras.layers import MaxPooling3D
from keras.layers import AveragePooling3D
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from keras.layers import GlobalAveragePooling3D

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.utils import plot_model
from SpectralNormalizationKeras import DenseSN, ConvSN1D, ConvSN2D, ConvSN3D


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

    x = ConvSN3D(filters, (3, 3, 3),
        strides=strides,
        padding=padding,
        use_bias=False)(x)

    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    x = Activation('relu')(x)

    return x

def grouped_convolution_block(input, grouped_channels, cardinality):

    init = input
    channel_axis = -1
    print(grouped_channels)
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = conv3d_bn(init, grouped_channels, strides=(1, 1, 1), padding='same')

        return x

    for c in range(cardinality):

        x = Lambda(lambda z: z[:, :, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        #print(x)

        #x = conv3d_bn(x, grouped_channels, strides=(1, 1, 1), padding='same')
        x = ConvSN3D(grouped_channels, (3, 3, 3), strides=(1,1,1), padding='same', use_bias=False)(x)

        group_list.append(x)

    group_merge = layers.concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)
    return x

def Inception_Inflated3d(img_input):
    channel_axis = -1
    cardinality = 8
    #Instantiates the Inflated 3D Inception v1 architecture.

    # Downsampling via convolution (spatial and temporal)
    x = conv3d_bn(img_input, 64, strides=(2, 2, 2), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_2a_3x3')(x)
    x = conv3d_bn(x, 64, strides=(1, 1, 1), padding='same')
    x = conv3d_bn(x, 192, strides=(1, 1, 1), padding='same')

    # Downsampling (spatial only)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same', name='MaxPool2d_3a_3x3')(x)

    # Mixed 3b
    branch_0 = conv3d_bn(x, 64,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn(x, 96, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 128, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn(x, 16,strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 32,strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 32, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3b')
    #print(x)
    
    # Mixed 3c
    branch_0 = conv3d_bn(x, 128, strides=(1, 1, 1),padding='same')

    branch_1 = conv3d_bn(x, 128,strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 192, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn(x, 32, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 96, strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_3c')
    
    grouped_channels = int(x._keras_shape[-1] / cardinality)
    print(grouped_channels)
    x = grouped_convolution_block(x, grouped_channels, cardinality)
    
    # Downsampling (spatial and temporal)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Mixed 4b
    branch_0 = conv3d_bn(x, 192, strides=(1, 1, 1),padding='same')

    branch_1 = conv3d_bn(x, 96, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 208,strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn(x, 16, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 48,strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4b')

    # Mixed 4c
    branch_0 = conv3d_bn(x, 160, strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn(x, 112, strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 224, strides=(1, 1, 1), padding='same')

    branch_2 = conv3d_bn(x, 24, strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 64, strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, strides=(1, 1, 1), padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4c')
    
    grouped_channels = int(x._keras_shape[-1] / cardinality)
    print(grouped_channels)
    x = grouped_convolution_block(x, grouped_channels, cardinality)
    
    # Mixed 4d
    branch_0 = conv3d_bn(x, 128,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn(x, 128, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 256, strides=(1, 1, 1),padding='same')

    branch_2 = conv3d_bn(x, 24, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 64, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 64, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4d')
#
#     Mixed 4e
#    branch_0 = conv3d_bn(x, 112,strides=(1, 1, 1), padding='same')
#
#    branch_1 = conv3d_bn(x, 144, strides=(1, 1, 1),padding='same')
#    branch_1 = conv3d_bn(branch_1, 288, padding='same')
#
#    branch_2 = conv3d_bn(x, 32,strides=(1, 1, 1), padding='same')
#    branch_2 = conv3d_bn(branch_2, 64, padding='same')
#
#    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
#    branch_3 = conv3d_bn(branch_3, 64, strides=(1, 1, 1),padding='same')
#
#    x = layers.concatenate(
#        [branch_0, branch_1, branch_2, branch_3],
#        axis=channel_axis,
#        name='Mixed_4e')

    # Mixed 4f
    branch_0 = conv3d_bn(x, 256,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn(x, 160, strides=(1, 1, 1),padding='same')
    branch_1 = conv3d_bn(branch_1, 320, padding='same')

    branch_2 = conv3d_bn(x, 32,strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 128, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_4f')

    # Downsampling (spatial and temporal)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')(x)
    
    grouped_channels = int(x._keras_shape[-1] / cardinality)
    print(grouped_channels)
    x = grouped_convolution_block(x, grouped_channels, cardinality)
    
    # Mixed 5b
    branch_0 = conv3d_bn(x, 256,strides=(1, 1, 1), padding='same')

    branch_1 = conv3d_bn(x, 160,strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 320, padding='same')

    branch_2 = conv3d_bn(x, 32, strides=(1, 1, 1), padding='same')
    branch_2 = conv3d_bn(branch_2, 128, strides=(1, 1, 1),padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, strides=(1, 1, 1),padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5b')

    # Mixed 5c
    branch_0 = conv3d_bn(x, 384, strides=(1, 1, 1),padding='same')

    branch_1 = conv3d_bn(x, 192, strides=(1, 1, 1), padding='same')
    branch_1 = conv3d_bn(branch_1, 384,strides=(1, 1, 1),  padding='same')

    branch_2 = conv3d_bn(x, 48, strides=(1, 1, 1),padding='same')
    branch_2 = conv3d_bn(branch_2, 128,strides=(1, 1, 1), padding='same')

    branch_3 = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    branch_3 = conv3d_bn(branch_3, 128, strides=(1, 1, 1), padding='same')

    x = layers.concatenate(
        [branch_0, branch_1, branch_2, branch_3],
        axis=channel_axis,
        name='Mixed_5c')
    #x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = GlobalAveragePooling3D()(x)
    return x

#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1],patchSize[2], 1))

    output = Inception_Inflated3d(img_input=input_tensor)

#    x = Dropout(0.5)(output)
#    x = Dense(1024)(x)
    x = Dropout(0.1)(output)
    x = Dense(512)(x)
#    x = Dropout(0.3)(x)
#    x = Dense(256)(x)
    x = Dropout(0.1)(x)
    x = Dense(128)(x)


    # fully-connected layer
    output = Dense(units=1,
                   activation='linear',
                   name='regression')(x)
#    output = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
#                  kernel_initializer='he_normal', activation='softmax')(x)
    sModelName = 'BioAgeNet_Regression'
    cnn = Model(input_tensor, output, name=sModelName)
    return cnn, sModelName

if __name__ == '__main__':
    model,_ = createModel([121,145, 121, 1])
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/inception_resnext_3d.png', show_shapes=True)
    model.summary()