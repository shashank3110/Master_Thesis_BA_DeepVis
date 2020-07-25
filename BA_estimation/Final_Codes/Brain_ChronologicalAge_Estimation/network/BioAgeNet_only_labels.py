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
        
        if c < (cardinality/2):
            x = Conv2D(grouped_channels, (1, 1), padding='same', use_bias=False, strides=(strides, strides),
                  kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        else:
            x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                 kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)

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
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if init._keras_shape[-1] != filters:
            print('Shape is diff, passing shortcut through 1x1 block')
            print(init._keras_shape[-1])
            init = Conv2D(filters, (1, 1), padding='same', strides=(strides, strides),
                          use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
            init = BatchNormalization(axis=channel_axis)(init)
    elif strides!=1:
            print('Strides not 1, Downsampling...')
            init = MaxPooling2D(pool_size=(2, 2))(init)
            init = BatchNormalization(axis=channel_axis)(init)

    x = Conv2D(grouped_channels, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = __grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)

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
        N = [(depth - 2) // 9 for _ in range(4)]
    
    filters = cardinality * width
    filters_list = []

    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters
    print(filters_list)
    x = __initial_conv_block(img_input, weight_decay)

   
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if (i == (n_i-1)) & (block_idx!=3):
                print('Entered last loop of the block')
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

def createModel(patchSize, num_slice_per_group, weight_decay, depth, cardinality, width):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], num_slice_per_group))


    def get_slice (x, index):
        return x[:, :, :, index]

    input_list = []

    # create the input tensor list
    for i in range(num_slice_per_group):
        sub_input = Lambda(get_slice, output_shape=(patchSize[0], patchSize[1], 1), arguments={'index': i})(
            input_tensor)
        sub_input = Reshape((patchSize[0], patchSize[1], 1))(sub_input)
        input_list.append(sub_input)
    print(input_list)
    # instance the basic model
    #basic_model = BasicModel()

    output_list =[]

    # use basic network for each input slice, and append their output
    for input in input_list:
        output = __create_res_next(weight_decay, depth, cardinality, width, img_input = input)
        output_list.append(output)
        print(output_list)
    # concatenate image and metadata

    x = concatenate([output_list[0], output_list[1], output_list[2]])

    x = Dropout(0.5)(x)
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
    model,_ = createModel([224,224], 3, weight_decay=4e-5, depth=[3, 3, 3, 3], cardinality = 16, width = 16)
    plot_model(model, to_file='/home/d1308/no_backup/d1308/BioAge_Regression_Network_ResNext/network/BioAge_Regression_Network.png', show_shapes=True)
    model.summary()
    
