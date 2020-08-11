# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 17:58:09 2019

@author: Anish
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model

CHANNEL_AXIS = -1

def _bn_relu(input):
    """Helper to build a BN -> relu block (by @raghakot)."""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def createModel(patchSize):

    input_tensor = Input(shape=(patchSize[0], patchSize[1], patchSize[2], 1))

    x = Conv3D(filters=8, kernel_size=(3, 3, 3),
                  strides=1, padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(5e-4))(input_tensor)
    x = _bn_relu(x)
    
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)
    
    x = Conv3D(filters=16, kernel_size=(3, 3, 3),
                  strides=1, padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(5e-4))(x)
    x = _bn_relu(x)
    
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)
    
    
    x = Conv3D(filters=32, kernel_size=(3, 3, 3),
                  strides=1, padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(5e-4))(x)
    
    x = _bn_relu(x)
    
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)
        
#    x = Conv3D(filters=32, kernel_size=(3, 3, 3),
#                  strides=1, padding="same",
#                  kernel_initializer="he_normal",
#                  kernel_regularizer=l2(1e-4))(x)
#    x = _bn_relu(x)
#    
#    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
#                             padding="valid")(x)
    
   
    x = Conv3D(filters= 64, kernel_size=(3, 3, 3),
                  strides=1, padding="same",
                  kernel_initializer="he_normal",
                  kernel_regularizer=l2(5e-4))(x)
    x = _bn_relu(x)
    
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                             padding="valid")(x)
    
    x = Flatten()(x)
  
    x = Dense(1024)(x)
    
    x = Dense(1024)(x)
   
    
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
    plot_model(model, to_file='/home/d1308/no_backup/d1308/3T_Brain_age_regression/network/papercode_2.png', show_shapes=True)
    model.summary()