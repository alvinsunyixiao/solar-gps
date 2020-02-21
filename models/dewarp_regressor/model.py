import numpy as np
import tensorflow as tf

from utils.params import ParamDict as o

K = tf.keras
KL = tf.keras.layers

class SameConvBNReLU(KL.Layer):

    def __init__(self, filters, kernel_size, strides,
                 weight_decay=None, has_bn=True, has_relu=True, **kwargs):
        super(SameConvBNReLU, self).__init__(**kwargs)
        regularizer = K.regularizers.l2(weight_decay) if weight_decay else None
        self.conv = KL.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=not has_bn,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizer,
            bias_initializer=K.initializers.constant(0),
            bias_regularizer=regularizer,
            trainable=self.trainable,
            name='conv',
        )
        self.bn = KL.BatchNormalization(
            axis=1,
            trainable=self.trainable,
            name='bn',
        ) if has_bn else None
        self.relu = KL.ReLU(name='relu') if has_relu else None

    def call(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x

class FullyConnected(KL.Layer):

    def __init__(self, units, weight_decay=None, has_relu=True, **kwargs):
        super(FullyConnected, self).__init__(**kwargs)
        regularizer = K.regularizers.l2(weight_decay) if weight_decay else None
        self.fc = KL.Dense(
            units=units,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            trainable=self.trainable,
            name='fc',
        )
        self.relu = KL.ReLU(name='relu') if has_relu else None

    def call(self, x):
        x = self.fc(x)
        if self.relu:
            x = self.relu(x)
        return x

class RegressorModel:

    DEFAULT_PARAMS=o(
        input_hw=(32, 512),
        batch_size=128,
        weight_decay=1e-4,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        assert K.backend.image_data_format() == 'channels_first', \
               'Image data format has to be channels first!'
        self.p = params
        self.convs = [
            SameConvBNReLU(8, (7,7), 2, weight_decay=self.p.weight_decay),
            SameConvBNReLU(8, (3,3), 1, weight_decay=self.p.weight_decay),
            SameConvBNReLU(8, (3,3), 2, weight_decay=self.p.weight_decay),
            SameConvBNReLU(8, (3,3), 1, weight_decay=self.p.weight_decay),
            SameConvBNReLU(16, (3,3), 2, weight_decay=self.p.weight_decay),
            SameConvBNReLU(16, (3,3), 1, weight_decay=self.p.weight_decay),
            SameConvBNReLU(32, (3,3), 2, weight_decay=self.p.weight_decay),
            SameConvBNReLU(32, (3,3), 1, weight_decay=self.p.weight_decay),
        ]
        self.fcs = [
            FullyConnected(256, weight_decay=self.p.weight_decay),
            FullyConnected(256, weight_decay=self.p.weight_decay),
            FullyConnected(1, weight_decay=self.p.weight_decay),
        ]

    def create_model(self):
        inp = KL.Input(shape=(1,) + self.p.input_hw, name='input_b1hw')
        x = inp
        for conv in self.convs:
            x = conv(x)
        x = KL.Reshape([np.prod(x.shape[1:])])(x)
        for fc in self.fcs:
            x = fc(x)
        return K.Model(inputs=[inp], outputs=[x])
