from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, warnings
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from keras.layers.core import Dense, Lambda

from keras.utils import plot_model


# Modular function for Fire Node

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution2D(squeeze, (1, 1), padding='valid')(x)
    x = Activation('relu')(x)

    left = Convolution2D(expand, (1, 1), padding='valid')(x)
    left = Activation('relu')(left)

    right = Convolution2D(expand, (3, 3), padding='same')(x)
    right = Activation('relu')(right)

    x = concatenate([left, right], axis=channel_axis)
    return x


# Original SqueezeNet from paper.

def __create_squeeze_net(img_input,pooling ='avg'):
    """Instantiates the SqueezeNet architecture.
    """


    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    #x = Dropout(0.4)(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    #x = Dropout(0.4)(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Dropout(0.4)(x)

   # x = fire_module(x, fire_id=6, squeeze=48, expand=192)
   # x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)

    x = Dropout(0.4)(x)

    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    

    
        #x = Dropout(0.5, name='drop9')(x)

        #x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        #x = Activation('relu', name='relu_conv10')(x)
        #x = GlobalAveragePooling2D()(x)
        #x = Activation('softmax', name='loss')(x)

    if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
    elif pooling=='max':
            x = GlobalMaxPooling2D()(x)

    return x

#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], 1))

    output = __create_squeeze_net(img_input=input_tensor)

    x = Dropout(0.3)(output)
    x = Dense(512)(x)
    x = Dropout(0.3)(x)
    x = Dense(256)(x)
    x = Dropout(0.3)(x)
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
    model,_ = createModel([224,224])
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/BioAge_Regression_SqueezeNetwork.png', show_shapes=True)
    model.summary()