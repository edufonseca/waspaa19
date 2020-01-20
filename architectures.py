
from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation, Permute, \
    Concatenate, Multiply, Reshape, GlobalAveragePooling2D
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np


def get_model_baseline(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # l1
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(24, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # l2
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # l3
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Flatten()(spec_x)
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(spec_x)

    spec_x = Dropout(0.5)(spec_x)
    out = Dense(n_class,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out)

    return model


def DenseNet_module_se(filter_size):
    def f(inputs):
        d_x = BatchNormalization()(inputs)
        d_x = Activation('relu')(d_x)
        d_x = Conv2D(filter_size, (1, 1), padding='same', activation='linear')(d_x)
        d_x = BatchNormalization()(d_x)
        d_x = Activation('relu')(d_x)
        d_x = Conv2D(filter_size, (3, 3), padding='same', activation='linear')(d_x)

        se = GlobalAveragePooling2D()(d_x)
        se = Reshape((1, 1, filter_size))(se)
        se = Dense(filter_size // 2, activation='relu')(se)
        se = Dense(filter_size, activation='sigmoid')(se)
        x = Multiply()([d_x, se])

        x = Concatenate()([x, inputs])
        return x
    return f


def get_model_DenSE(params_learn=None, params_extract=None):
    """
    based on:
    https://github.com/finejuly/dcase2018_task2_cochlearai/blob/master/utils/model.py
    """

    K.set_image_data_format('channels_last')
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Permute((2, 3, 1))(spec_x)
    spec_x1 = Conv2D(15, (3, 3), padding='same', activation='linear')(spec_x)
    spec_x = Concatenate()([spec_x, spec_x1])

    # DenseNet modules + SE
    for i in range(params_learn.get('densenet_modules')):
        spec_x = DenseNet_module_se(np.minimum(512, 16*(2**i)))(spec_x)
        spec_x = MaxPooling2D(pool_size=(2, 2), padding='same')(spec_x)
        # spec_x = Dropout(0.5)(spec_x)

    spec_x = BatchNormalization()(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = Flatten()(spec_x)
    spec_x = Dropout(0.5)(spec_x)
    outputs = Dense(n_class,
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-3),
                    activation='softmax',
                    name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=outputs)
    return model
