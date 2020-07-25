from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Lambda
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers.merge import add, concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model


# Modular function for Fire Node
def _bn_relu(input):
    CHANNEL_AXIS = -1
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

def fire_module(x, squeeze=16, expand=64):

    channel_axis = -1

    x = Conv3D(squeeze, (1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    x = _bn_relu(x)

    left = Conv3D(expand, (1, 1, 1), padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)
    left = Activation('relu')(left)

    right = Conv3D(expand, (3, 3, 3), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(x)

    right = Activation('relu')(right)

    x = concatenate([left, right], axis=channel_axis)
    return x


# Original SqueezeNet from paper.

def __create_squeeze_net(img_input,pooling ='avg'):
    """Instantiates the SqueezeNet architecture.
    """


    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2),padding='valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(5e-4))(img_input)
    x = _bn_relu(x)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                     padding="valid")(x)
    #x = Dropout(0.4)(x)

    x = fire_module(x, squeeze=16, expand=64)
    x = fire_module(x, squeeze=16, expand=64)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                     padding="valid")(x)
    #x = Dropout(0.4)(x)

    x = fire_module(x, squeeze=32, expand=128)
    x = fire_module(x, squeeze=32, expand=128)
    x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2),
                     padding="valid")(x)
    #x = Dropout(0.4)(x)

   # x = fire_module(x, fire_id=6, squeeze=48, expand=192)
   # x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, squeeze=64, expand=256)

    #x = Dropout(0.4)(x)

    x = fire_module(x, squeeze=64, expand=256)
    

    
        #x = Dropout(0.5, name='drop9')(x)

        #x = Convolution2D(classes, (1, 1), padding='valid', name='conv10')(x)
        #x = Activation('relu', name='relu_conv10')(x)
        #x = GlobalAveragePooling2D()(x)
        #x = Activation('softmax', name='loss')(x)

    if pooling == 'avg':
            x = GlobalAveragePooling3D()(x)
    elif pooling=='max':
            x = GlobalMaxPooling2D()(x)

    return x

#########################
# Input: patchSize (image size of the slice in the group)
# Output: AgeNet model
# num_slice_per_group: the number of slices are extracted in a sample
##########################

def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1],patchSize[2], 1))

    output = __create_squeeze_net(img_input=input_tensor)

    #x = Dropout(0.3)(output)
    x = Dense(256)(output)
    #x = Dropout(0.3)(x)
    x = Dense(256)(x)
    #x = Dropout(0.3)(x)
    #x = Dense(128)(x)

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
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/squeeze_3D.png', show_shapes=True)
    model.summary()