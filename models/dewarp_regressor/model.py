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
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DeWarp(KL.Layer):

    def __init__(self, dewarp_hw, radius_range, **kwargs):
        super(DeWarp, self).__init__(**kwargs)
        self.dewarp_hw = dewarp_hw
        self.radius_range = radius_range

    def raw2aop(self, image_b1hw):
        """ convert raw sensor image to AoP

        Args:
            image_b1hw (4-D tensor):  raw sensor image

        Returns:
            AoP
        """
        image_b1hw = tf.cast(image_b1hw, tf.float32)
        i0_b1hw = image_b1hw[..., ::2, ::2]
        i45_b1hw = image_b1hw[..., ::2, 1::2]
        i90_b1hw = image_b1hw[..., 1::2, 1::2]
        i135_b1hw = image_b1hw[..., 1::2, ::2]

        s0_b1hw = i0_b1hw + i90_b1hw
        s1_b1hw = i0_b1hw - i90_b1hw
        s2_b1hw = i45_b1hw - i135_b1hw

        aop_b1hw = .5 * tf.atan2(s2_b1hw, s1_b1hw)

        return aop_b1hw

    def transform(self, polar_b1hw):
        H, W = self.dewarp_hw
        Rmin, Rmax = self.radius_range
        y_h = tf.range(H, dtype=tf.float32)
        x_w = tf.range(W, dtype=tf.float32)
        y_hw, x_hw = tf.meshgrid(y_h, x_w, indexing='ij')
        # convert to polar coordinates
        r_hw = (Rmax - Rmin) / H * (H - y_hw) + Rmin
        theta_hw = 2 * np.pi / W * (W - x_hw)
        # random rotation augmentation
        theta_hw += tf.random.uniform([], 0, 2*np.pi)
        # put back into cartesian coordinates of the fisheye image
        polar_shape = tf.cast(tf.shape(polar_b1hw), tf.float32)
        polar_y_hw = -r_hw * tf.sin(theta_hw) + polar_shape[2] // 2
        polar_x_hw = r_hw * tf.cos(theta_hw) + polar_shape[3] // 2
        # round to nearest neighbor
        polar_y_hw = tf.cast(tf.round(polar_y_hw), tf.int32)
        polar_x_hw = tf.cast(tf.round(polar_x_hw), tf.int32)
        # calculate batched coordinate lookup
        channel_0_hw = tf.zeros_like(polar_y_hw)
        polar_0yx_hw3 = tf.stack([channel_0_hw, polar_y_hw, polar_x_hw], axis=-1)
        polar_0yx_bhw3 = tf.tile(polar_0yx_hw3[tf.newaxis, ...],
                                 [polar_shape[0], 1, 1, 1])
        batch_b_b = tf.range(tf.shape(polar_b1hw)[0])
        batch_b_b111 = batch_b_b[:, tf.newaxis, tf.newaxis, tf.newaxis]
        batch_b_bhw1 = tf.tile(batch_b_b111, [1, H, W, 1])
        polar_b0yx_bhw4 = tf.concat([batch_b_bhw1, polar_0yx_bhw3], axis=-1)
        # perform the transformation lookup
        dewarp_bhw = tf.gather_nd(polar_b1hw, polar_b0yx_bhw4)
        dewarp_b1hw = tf.expand_dims(dewarp_bhw, axis=1)
        return dewarp_b1hw

    def call(self, raw_image_b1hw):
        aop_b1hw = self.raw2aop(raw_image_b1hw)
        dewarp_b1hw = self.transform(aop_b1hw)
        return dewarp_b1hw


class ResBlock(KL.Layer):

    def __init__(self, filters, strided=False, projection=False,
                 weight_decay=None, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        if strided:
            projection = True
        self.conv1 = SameConvBNReLU(
            filters=filters,
            kernel_size=3,
            strides=2 if strided else 1,
            weight_decay=weight_decay,
            trainable=self.trainable,
            name='conv1',
        )
        self.conv2 = SameConvBNReLU(
            filters=filters,
            kernel_size=3,
            strides=1,
            weight_decay=weight_decay,
            trainable=self.trainable,
            name='conv2',
        )
        self.projection = SameConvBNReLU(
            filters=filters,
            kernel_size=1,
            strides=2 if strided else 1,
            weight_decay=weight_decay,
            trainable=self.trainable,
            name='projection',
        ) if projection else None

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        if self.projection is not None:
            x = self.projection(x)
        return x + x1

class RegressorModel:

    DEFAULT_PARAMS=o(
        input_hw=(2048, 2448),
        batch_size=128,
        weight_decay=1e-4,
        dewarp_hw=(32, 512),
        radius_range=(280, 400),
    )

    def __init__(self, params=DEFAULT_PARAMS):
        assert K.backend.image_data_format() == 'channels_first', \
               'Image data format has to be channels first!'
        self.p = params
        self.dewarp = DeWarp(self.p.dewarp_hw, self.p.radius_range)
        self.conv1 = SameConvBNReLU(8, (7, 7), 2, weight_decay=self.p.weight_decay)
        self.pool2 = KL.MaxPool2D(strides=2, padding='same')
        self.res_blocks = [
            ResBlock(8, projection=True, weight_decay=self.p.weight_decay),
            ResBlock(8, weight_decay=self.p.weight_decay),
            ResBlock(16, strided=True, weight_decay=self.p.weight_decay),
            ResBlock(16, weight_decay=self.p.weight_decay),
            ResBlock(32, strided=True, weight_decay=self.p.weight_decay),
            ResBlock(32, weight_decay=self.p.weight_decay),
            ResBlock(64, strided=True, weight_decay=self.p.weight_decay),
            ResBlock(64, weight_decay=self.p.weight_decay),
        ]
        self.reduce = KL.Dense(
            units=1,
            kernel_initializer=K.initializers.RandomNormal(stddev=1e-2),
        )

    def create_model(self):
        inp = KL.Input(shape=(1,) + self.p.input_hw)
        x = self.dewarp(inp)
        x = self.conv1(x)
        x = self.pool2(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        x = KL.Reshape([np.prod(x.shape[1:])])(x)
        x = self.reduce(x)

        return K.Model(inputs=[inp], outputs=[x])
