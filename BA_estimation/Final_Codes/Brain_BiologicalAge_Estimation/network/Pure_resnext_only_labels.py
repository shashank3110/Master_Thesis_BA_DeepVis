'''ResNeXt models for Keras.
# Reference
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from keras.utils import plot_model
from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D
from keras.layers import Input, Dropout, Reshape
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K


def __initial_conv_block(input, weight_decay=5e-4):
    ''' Adds an initial convolution block, with batch normalization and relu activation
    Args:
        input: input tensor
        weight_decay: weight decay factor
    Returns: a keras tensor
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    return x


def __grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    ''' Adds a grouped convolution block. It is an equivalent block from the paper
    Args:
        input: input tensor
        grouped_channels: grouped number of filters
        cardinality: cardinality factor describing the number of groups
        strides: performs strided convolution for downscaling if > 1
        weight_decay: weight decay term
    Returns: a keras tensor
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels]
        if K.image_data_format() == 'channels_last' else
        lambda z: z[:, c * grouped_channels:(c + 1) * grouped_channels, :, :])(input)

        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                  kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
#        x = Conv2D(grouped_channels, (1, 1), padding='same', use_bias=False, strides=(strides, strides),
#                 kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
#        x = Conv2D(grouped_channels, (1, 3), padding='same', use_bias=False, strides=1,
#                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
#        x = Conv2D(grouped_channels, (3, 1), padding='same', use_bias=False, strides=1,
#                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=channel_axis)
    x = BatchNormalization(axis=channel_axis)(group_merge)
    x = Activation('relu')(x)

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
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 2 * filters:
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    else:
        print('entered1')
        if init._keras_shape[-1] != 2 * filters:
            print('entered2')
            print(init._keras_shape[-1])
            init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=channel_axis)(x)

    x = add([init, x])
    x = Activation('relu')(x)

    return x


def __create_res_next(weight_decay, depth, cardinality, width,
                       img_input):
    ''' Creates a ResNeXt model with specified parameters
    Args:
        nb_classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        depth: Depth of the network. Can be an positive integer or a list
               Compute N = (n - 2) / 9.
               For a depth of 56, n = 56, N = (56 - 2) / 9 = 6
               For a depth of 101, n = 101, N = (101 - 2) / 9 = 11
        cardinality: the size of the set of transformations.
               Increasing cardinality improves classification accuracy,
        width: Width of the network.
        weight_decay: weight_decay (l2 norm)
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
    Returns: a Keras Model
    '''

    if type(depth) is list or type(depth) is tuple:
        # If a list is provided, defer to user how many blocks are present
        N = list(depth)
    else:
        # Otherwise, default to 3 blocks each of default number of group convolution blocks
        # for 3 times calculate depth-2/9
        # So results in N = [6 6 6]
        N = [(depth - 2) // 9 for _ in range(3)]
        print(N)
    print(len(N))
    filters = cardinality * width
    print(filters)
    filters_list = []

    # len(N) = 3
    # filters_list = [128 256 512]
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters
    print(filters_list)
    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    # N[0] = 6 so 6 blocks
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    # N = [6 6]
    filters_list = filters_list[1:]  # remove the first filter from the filter list

    print(filters_list)
    print(N)
    print(len(N))
    # block 2 to N
    # Now N = [6 6]
    # len(N) = 2
    # block_idx takes 2 values that is 0,1 because it is the length/ total values present in N, N has [6 6]
    # n_i takes two values that is 6 and 6
    # first iteration for block_idx = 0, n_i presumes value of 6 and thus 6 blocks are created with filter[0] which is now 256
    # second iteration for block idx = 1, n_i again presumes 6 and thus 6 blocks are created with filter[1] which is now 512
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                print('i==0')
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)
                #x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)    

    return x


#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize, weight_decay, depth, cardinality, width):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], 1))

    output = __create_res_next(weight_decay, depth, cardinality, width, img_input=input_tensor)

    #x = concatenate([output_list[0], output_list[1], output_list[2]])

    x = Dropout(0.5)(output)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    x = Dense(256)(x)
    x = Dropout(0.5)(x)
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
    model,_ = createModel([224,224], weight_decay=4e-5, depth=[1, 1, 1], cardinality = 16, width = 8)
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/Pure_ResNext_labels_only.png', show_shapes=True)
    model.summary()
    
