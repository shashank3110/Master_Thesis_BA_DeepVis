'''ResNeXt models for Keras.
# Reference
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Lambda,
    Dropout,
GlobalAveragePooling3D
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model

CHANNEL_AXIS = -1

def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def __initial_conv_block(input, weight_decay=1e-4):

    channel_axis = -1

    x = Conv3D(8, (3, 3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)



    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                     padding="valid")(x)

    return x


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):

    init = input
    channel_axis = -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
   
        x = Lambda(lambda z: z[:, :, :,:, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv3D(grouped_channels, (3, 3, 3), padding='same', use_bias=False, strides=(strides, strides, strides),
                  kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)

        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)

    return x


def __bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    ''' Adds a bottleneck block
    Args:
        input: input tensor
        filters: number of output filters
        cardinality: cardinality factor described number of
            grouped convolutions
        strides: performs strided convolution for downsampling if > 1
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    init = input
    print('The number of output filters are')
    print(filters)
    grouped_channels = int(filters / cardinality)
    channel_axis = -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init._keras_shape[-1] != 2 * filters:
            print('entered2')
            print(init._keras_shape[-1])
            init = Conv3D(filters * 2, (1, 1, 1), padding='same', strides=(strides, strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv3D(filters, (1, 1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    print(x)
    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    print(x)
    x = Conv3D(filters * 2, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)
    print(x)
    x = add([init, x])
    x = Activation('relu')(x)

    return x


def __create_res_next(weight_decay=1e-4, depth = [1, 1, 1], cardinality=8, width=16, img_input = None):

    N = depth
    filters = cardinality * width
    print(filters)
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters
    print(filters_list)
    x = __initial_conv_block(img_input, weight_decay)

    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)
        
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)

    N = N[1:]  # remove the first block from block definition list
#    N = [6 6]
    filters_list = filters_list[1:]  # remove the first filter from the filter list
    
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                print('i==0')
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)

    return x

#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize, weight_decay, depth, cardinality, width):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], patchSize[2], 1))   
    
    x = __create_res_next(weight_decay, depth, cardinality, width, img_input = input_tensor)
    #print(x)
    #x = Conv3D(filters=32, kernel_size=(3, 3, 3),
    #           strides=1, padding="same",
    #           kernel_initializer="he_normal",
    #           kernel_regularizer=l2(1e-4))(x)

    #x = _bn_relu(x)

    #x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
    #                 padding="valid")(x)
    #x = Flatten()(x)
    x = GlobalAveragePooling3D()(x)
    x = Dropout(0.1)(x)
    x = Dense(units=512,
                          kernel_initializer="he_normal",
                          activation='relu',
                          kernel_regularizer=l2(5e-4))(x)
    x = Dropout(0.1)(x)
    x = Dense(units=128,
                          kernel_initializer="he_normal",
                          activation='relu',
                          kernel_regularizer=l2(5e-4))(x)

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
    model,_ = createModel([121, 145, 121, 1], weight_decay=5e-4, depth=[2,2,2], cardinality = 8, width = 8)
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/papercode_3.png', show_shapes=True)
    model.summary()
    
