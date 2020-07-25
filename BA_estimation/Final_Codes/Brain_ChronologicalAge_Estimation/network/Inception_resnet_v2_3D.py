"""Inception-ResNet V2 model for Keras.
Model naming and structure follows TF-slim implementation
(which has some additional layers and different number of
filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py
Pre-trained ImageNet weights are also converted from TF-slim,
which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models
# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261) (AAAI 2017)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.layers import Input, Dropout, Reshape
from keras.layers.core import Activation
from keras.layers.core import Dense, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import GlobalAveragePooling3D, GlobalMaxPooling3D, AveragePooling3D, MaxPooling3D
from keras.layers.merge import concatenate, add
from keras.utils import plot_model
from keras.models import Model
import os
import keras.backend as K


def conv3d_bn(x,
              filters,
              kernel_size,
              strides=(1, 1, 1),
              padding='same',
              activation = 'relu',
              use_bias = False
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

    x = Conv3D(filters, kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias)(x)

    bn_axis = -1
    x = BatchNormalization(axis=bn_axis, scale=False)(x)
    if activation is not None:
        x = Activation('relu')(x)

    return x

def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    """Adds a Inception-ResNet block.
    This function builds 3 types of Inception-ResNet blocks mentioned
    in the paper, controlled by the `block_type` argument (which is the
    block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`
    # Arguments
        x: input tensor.
        scale: scaling factor to scale the residuals (i.e., the output of
            passing `x` through an inception module) before adding them
            to the shortcut branch.
            Let `r` be the output from the residual branch,
            the output of this block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines
            the network structure in the residual branch.
        block_idx: an `int` used for generating layer names.
            The Inception-ResNet blocks
            are repeated many times in this network.
            We use `block_idx` to identify
            each of the repetitions. For example,
            the first Inception-ResNet-A block
            will have `block_type='block35', block_idx=0`,
            and the layer names will have
            a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block
            (see [activations](../activations.md)).
            When `activation=None`, no activation is applied
            (i.e., "linear" activation: `a(x) = x`).
    # Returns
        Output tensor for the block.
    # Raises
        ValueError: if `block_type` is not one of `'block35'`,
            `'block17'` or `'block8'`.
    """
    if block_type == 'block35':
        branch_0 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(x, 32, 1)
        branch_1 = conv3d_bn(branch_1, 32, 3)
        branch_2 = conv3d_bn(x, 32, 1)
        branch_2 = conv3d_bn(branch_2, 48, 3)
        branch_2 = conv3d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 128, 1)
        branch_1 = conv3d_bn(branch_1, 160, [1, 1, 7])
        branch_1 = conv3d_bn(branch_1, 192, [7, 1, 1])
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(x, 192, 1)
        branch_1 = conv3d_bn(branch_1, 224, [1, 3, 1])
        branch_1 = conv3d_bn(branch_1, 256, [3, 1, 1])
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    #block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    mixed = concatenate(branches)
    up = conv3d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True)

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=K.int_shape(x)[1:],
                      arguments={'scale': scale})([x, up])
    if activation is not None:
        x = Activation(activation)(x)
    return x


def InceptionResNetV2(img_input, pooling=None):
    """Instantiates the Inception-ResNet v2 architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is `False` (otherwise the input shape
            has to be `(299, 299, 3)` (with `'channels_last'` data format)
            or `(3, 299, 299)` (with `'channels_first'` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 75.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the last convolutional block.
            - `'avg'` means that global average pooling
                will be applied to the output of the
                last convolutional block, and thus
                the output of the model will be a 2D tensor.
            - `'max'` means that global max pooling will be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is `True`, and
            if no `weights` argument is specified.
    # Returns
        A Keras `Model` instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    # Stem block: 35 x 35 x 192
    x = conv3d_bn(img_input, 32, 3, strides=2, padding='valid')
    x = conv3d_bn(x, 32, 3, padding='valid')
    x = conv3d_bn(x, 64, 3)
    x = MaxPooling3D(3, strides=2)(x)
    x = conv3d_bn(x, 80, 1, padding='valid')
    x = conv3d_bn(x, 192, 3, padding='valid')
    x = MaxPooling3D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv3d_bn(x, 96, 1)
    branch_1 = conv3d_bn(x, 48, 1)
    branch_1 = conv3d_bn(branch_1, 64, 5)
    branch_2 = conv3d_bn(x, 64, 1)
    branch_2 = conv3d_bn(branch_2, 96, 3)
    branch_2 = conv3d_bn(branch_2, 96, 3)
    branch_pool = AveragePooling3D(3, strides=1, padding='same')(x)
    branch_pool = conv3d_bn(branch_pool, 64, 1)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = concatenate(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 2):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv3d_bn(x, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 256, 3)
    branch_1 = conv3d_bn(branch_1, 384, 3, strides=2, padding='valid')
    branch_pool = MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = concatenate(branches)

    # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 2):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv3d_bn(x, 256, 1)
    branch_0 = conv3d_bn(branch_0, 384, 3, strides=2, padding='valid')
    branch_1 = conv3d_bn(x, 256, 1)
    branch_1 = conv3d_bn(branch_1, 288, 3, strides=2, padding='valid')
    branch_2 = conv3d_bn(x, 256, 1)
    branch_2 = conv3d_bn(branch_2, 288, 3)
    branch_2 = conv3d_bn(branch_2, 320, 3, strides=2, padding='valid')
    branch_pool = MaxPooling3D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = concatenate(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 2):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10)

    # Final convolution block: 8 x 8 x 1536
    x = conv3d_bn(x, 1536, 1)

    if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
    elif pooling == 'max':
            x = GlobalMaxPooling3D()(x)

    return x

#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], patchSize[2], 1))


    #def get_slice (x, index):
    #    return x[:, :, :, index]

    #input_list = []

    # create the input tensor list
    #for i in range(num_slice_per_group):
    #    sub_input = Lambda(get_slice, output_shape=(patchSize[0], patchSize[1], 1), arguments={'index': i})(
    #        input_tensor)
    #    sub_input = Reshape((patchSize[0], patchSize[1], 1))(sub_input)
    #    input_list.append(sub_input)
    #print(input_list)
    # instance the basic model
    #basic_model = BasicModel()

    #output_list =[]

    # use basic network for each input slice, and append their output
    #for input in input_list:
    #    output = InceptionResNetV2(img_input = input, pooling='avg')
    #    output_list.append(output)
    #    print(output_list)
    # concatenate image and metadata

    #x = concatenate([output_list[0], output_list[1], output_list[2]])
    output = InceptionResNetV2(img_input=input_tensor, pooling='avg')
    x = Dropout(0.1)(output)
    x = Dense(512)(x)
    x = Dropout(0.1)(x)
    #x = Dense(256)(x)
    #x = Dropout(0.5)(x)
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
    model,_ = createModel([121, 145, 121, 1])
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/Inception_resnet_3d.png', show_shapes=True)
    model.summary()