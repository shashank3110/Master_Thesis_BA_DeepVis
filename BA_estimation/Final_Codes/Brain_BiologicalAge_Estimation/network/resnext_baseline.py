'''ResNeXt models for Keras.
# Reference
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf))
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings
from keras.utils import plot_model
from keras.models import Model
from keras.layers.core import Dense, Lambda
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras_applications.imagenet_utils import _obtain_input_shape
import keras.backend as K

#CIFAR_TH_WEIGHTS_PATH = ''
#CIFAR_TF_WEIGHTS_PATH = ''
#CIFAR_TH_WEIGHTS_PATH_NO_TOP = ''
#CIFAR_TF_WEIGHTS_PATH_NO_TOP = ''
#
#IMAGENET_TH_WEIGHTS_PATH = ''
#IMAGENET_TF_WEIGHTS_PATH = ''
#IMAGENET_TH_WEIGHTS_PATH_NO_TOP = ''
#IMAGENET_TF_WEIGHTS_PATH_NO_TOP = ''


def ResNext(input_shape=None, depth=29, cardinality=8, width=16, weight_decay=5e-4,
            include_top=True, weights=None, input_tensor=None,
            pooling=None, classes=10):
    """Instantiate the ResNeXt architecture. Note that ,
        when using TensorFlow for best performance you should set
        `image_data_format="channels_last"` in your Keras config
        at ~/.keras/keras.json.
        The model are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            depth: number or layers in the ResNeXt model. Can be an
                integer or a list of integers.
            cardinality: the size of the set of transformations
            width: multiplier to the ResNeXt width (number of filters)
            weight_decay: weight decay (l2 norm)
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(32, 32, 3)` (with `tf` dim ordering)
                or `(3, 32, 32)` (with `th` dim ordering).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 8.
                E.g. `(200, 200, 3)` would be one valid value.
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
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Returns
            A Keras model instance.
        """

    if weights not in {'cifar10', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `cifar10` '
                         '(pre-training on CIFAR-10).')

    if weights == 'cifar10' and include_top and classes != 10:
        raise ValueError('If using `weights` as CIFAR 10 with `include_top`'
                         ' as true, `classes` should be 10')

    if type(depth) == int:
        if (depth - 2) % 9 != 0:
            raise ValueError('Depth of the network must be such that (depth - 2)'
                             'should be divisible by 9.')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=8,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_res_next(classes, img_input, include_top, depth, cardinality, width,
                          weight_decay, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnext')

#    # load weights
#    if weights == 'cifar10':
#        if (depth == 29) and (cardinality == 8) and (width == 64):
#            # Default parameters match. Weights for this model exist:
#
#            if K.image_data_format() == 'channels_first':
#                if include_top:
#                    weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels.h5',
#                                            CIFAR_TH_WEIGHTS_PATH,
#                                            cache_subdir='models')
#                else:
#                    weights_path = get_file('resnext_cifar_10_8_64_th_dim_ordering_th_kernels_no_top.h5',
#                                            CIFAR_TH_WEIGHTS_PATH_NO_TOP,
#                                            cache_subdir='models')
#
#                model.load_weights(weights_path)
#
#                if K.backend() == 'tensorflow':
#                    warnings.warn('You are using the TensorFlow backend, yet you '
#                                  'are using the Theano '
#                                  'image dimension ordering convention '
#                                  '(`image_dim_ordering="th"`). '
#                                  'For best performance, set '
#                                  '`image_dim_ordering="tf"` in '
#                                  'your Keras config '
#                                  'at ~/.keras/keras.json.')
#                    convert_all_kernels_in_model(model)
#            else:
#                if include_top:
#                    weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels.h5',
#                                            CIFAR_TF_WEIGHTS_PATH,
#                                            cache_subdir='models')
#                else:
#                    weights_path = get_file('resnext_cifar_10_8_64_tf_dim_ordering_tf_kernels_no_top.h5',
#                                            CIFAR_TF_WEIGHTS_PATH_NO_TOP,
#                                            cache_subdir='models')
#
#                model.load_weights(weights_path)
#
#                if K.backend() == 'theano':
#                    convert_all_kernels_in_model(model)

    return model


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


def __create_res_next(nb_classes, img_input, include_top, depth=29, cardinality=8, width=16,
                      weight_decay=5e-4, pooling=None):
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
        #So results in N = [6 6 6]
        N = [(depth - 2) // 9 for _ in range(3)]
        print(N)
    print(len(N))
    filters = cardinality * width
    print(filters)
    filters_list = []

#len(N) = 3
    #filters_list = [128 256 512]
    for i in range(len(N)):
        filters_list.append(filters)
        filters *= 2  # double the size of the filters
    print(filters_list)
    x = __initial_conv_block(img_input, weight_decay)

    # block 1 (no pooling)
    #N[0] = 6 so 6 blocks
    for i in range(N[0]):
        x = __bottleneck_block(x, filters_list[0], cardinality, strides=1, weight_decay=weight_decay)

    N = N[1:]  # remove the first block from block definition list
    #N = [6 6]
    filters_list = filters_list[1:]  # remove the first filter from the filter list
    
    print(filters_list)
    print(N)
    print(len(N))
    # block 2 to N
    #Now N = [6 6]
    #len(N) = 2
    #block_idx takes 2 values that is 0,1 because it is the length/ total values present in N, N has [6 6]
    #n_i takes two values that is 6 and 6
    #first iteration for block_idx = 0, n_i presumes value of 6 and thus 6 blocks are created with filter[0] which is now 256
    #second iteration for block idx = 1, n_i again presumes 6 and thus 6 blocks are created with filter[1] which is now 512
    for block_idx, n_i in enumerate(N):
        for i in range(n_i):
            if i == 0:
                print('i==0')
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=2,
                                       weight_decay=weight_decay)
            else:
                x = __bottleneck_block(x, filters_list[block_idx], cardinality, strides=1,
                                       weight_decay=weight_decay)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
                  kernel_initializer='he_normal', activation='softmax')(x)
#        x = Dense(1024, use_bias=False, kernel_regularizer=l2(weight_decay),
#                  kernel_initializer='he_normal', activation='softmax')(x)
#        x = Dense(nb_classes, use_bias=False, kernel_regularizer=l2(weight_decay),
#                  kernel_initializer='he_normal', activation='softmax')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x


if __name__ == '__main__':
    model = ResNext((304, 304, 3), depth = [2,2,2], cardinality=8, width=16)
    plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
    model.summary()
    
