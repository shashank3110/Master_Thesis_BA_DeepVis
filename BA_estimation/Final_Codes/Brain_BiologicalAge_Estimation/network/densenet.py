#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import preprocess_input as _preprocess_input
import keras.backend as K
from keras.utils import plot_model



def DenseNet(img_input,
             depth=40,
             nb_dense_block=3,
             growth_rate=12,
             nb_filter=-1,
             nb_layers_per_block=-1,
             bottleneck=False,
             reduction=0.0,
             dropout_rate=0.0,
             weight_decay=1e-4,
             subsample_initial_block=True,
             transition_pooling='avg'):


    x = __create_dense_net(img_input, depth, nb_dense_block,
                           growth_rate, nb_filter, nb_layers_per_block, bottleneck,
                           reduction, dropout_rate, weight_decay,subsample_initial_block,transition_pooling)

    return x



def __conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, block_prefix=None):


    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal',
                       padding='same', use_bias=False,
                       kernel_regularizer=l2(weight_decay))(x)
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
            x = Activation('relu')(x)

        x = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False,
                  dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True,
                  return_concat_list=False, block_prefix=None):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones
    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming
    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor
        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
            x_list.append(cb)

            x = concatenate([x, cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_block(ip, nb_filter, compression=1.0, weight_decay=1e-4,
                       block_prefix=None, transition_pooling='max'):
   
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(ip)
        x = Activation('relu')(x)
        x = Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal',
                   padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        if transition_pooling == 'avg':
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        elif transition_pooling == 'max':
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)

        return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4,
                          block_prefix=None):
    
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling2D()(ip)
        
        else:
            x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='same',
                                strides=(2, 2), kernel_initializer='he_normal',
                                kernel_regularizer=l2(weight_decay))(ip)
        return x


def __create_dense_net(img_input, depth=40, nb_dense_block=3,
                       growth_rate=12, nb_filter=-1, nb_layers_per_block=-1,
                       bottleneck=False, reduction=0.0, dropout_rate=None,
                       weight_decay=1e-4, subsample_initial_block=True, transition_pooling='avg'):
    
    with K.name_scope('DenseNet'):
        concat_axis = -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != nb_dense_block:
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, ('Depth must be 3 N + 4 '
                                              'if nb_layers_per_block == -1')
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (3, 3)
            initial_strides = (1, 1)

        x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal',
                   padding='same', name='initial_conv2D', strides=initial_strides,
                   use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5,
                                   name='initial_bn')(x)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter,
                                         growth_rate, bottleneck=bottleneck,
                                         dropout_rate=dropout_rate,
                                         weight_decay=weight_decay,
                                         block_prefix='dense_%i' % block_idx)
            # add transition_block
            x = __transition_block(x, nb_filter, compression=compression,
                                   weight_decay=weight_decay,
                                   block_prefix='tr_%i' % block_idx,
                                   transition_pooling=transition_pooling)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, growth_rate,
                                     bottleneck=bottleneck, dropout_rate=dropout_rate,
                                     weight_decay=weight_decay,
                                     block_prefix='dense_%i' % (nb_dense_block - 1))

        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5, name='final_bn')(x)
        x = Activation('relu')(x)

       
        return x

def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], 1))

    output = DenseNet(img_input=input_tensor, depth=8, nb_dense_block=4, growth_rate=32,
                    nb_filter=64, nb_layers_per_block=[3, 3, 3, 3],
                    bottleneck=True, reduction=0.5,
                    dropout_rate=0.0, weight_decay=1e-4,subsample_initial_block=True,transition_pooling = 'avg'
                    )

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
    model,_ = createModel([121,145, 1])
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/densenet.png', show_shapes=True)
    model.summary()