import math
import os

import tensorflow as tf

from utils.params import ParamDict as o

class DewarpEncoder:
    """ an encoder that performs circular de-warp operation """

    DEFAULT_PARAMS=o(
        # spatial dimension of the network input
        input_hw=(32, 512),
        # min / max radius of the original fisheye image
        radius_range=(280, 400),
        # minimum DoLP to be considered valid region
        min_intensity=1e-2,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params

    def raw2polarize(self, image):
        """ convert raw sensor image to AoP and DoLP

        Args:
            image (3-D TF Tensor):  raw sensor image

        Returns:
            Aop, DoLP
        """
        image = tf.cast(image, tf.float32)
        i0 = image[::2, ::2, 0]
        i45 = image[::2, 1::2, 0]
        i90 = image[1::2, 1::2, 0]
        i135 = image[1::2, ::2, 0]

        s0 = i0 + i90
        s1 = i0 - i90
        s2 = i45 - i135

        dop = tf.math.divide_no_nan(tf.sqrt(s1**2 + s2**2), s0)
        aop = .5 * tf.atan2(s2, s1)

        return aop, dop

    def dewarp(self, aop):
        """ dewarp the fisheye camera into a rectangular image by taking
        the outter ring area of the original image

        Args:
            aop (3-D TF Tensor):    angle of polarization image

        Returns:
            dewarped image
        """
        H, W = self.p.input_hw
        Rmin, Rmax = self.p.radius_range
        y_h = tf.range(H, dtype=tf.float32)
        x_w = tf.range(W, dtype=tf.float32)
        y_hw, x_hw = tf.meshgrid(y_h, x_w, indexing='ij')
        # convert to polar coordinate
        r_hw = (Rmax - Rmin) / H * (H - y_hw) + Rmin
        theta_hw = 2 * math.pi / W * (W - x_hw) + tf.random.uniform([], 0, 2*math.pi)
        # back to cartesian in old fisheye coordinates
        aop_shape = tf.cast(tf.shape(aop), tf.float32)
        circ_y_hw = -r_hw * tf.sin(theta_hw) + aop_shape[0] // 2
        circ_x_hw = r_hw * tf.cos(theta_hw) + aop_shape[1] // 2
        circ_yx_hw2 = tf.stack([circ_y_hw, circ_x_hw], axis=-1)
        # round to nearest neighbor
        circ_yx_hw2 = tf.cast(tf.round(circ_yx_hw2), tf.int32)
        dewarp_img_hw = tf.gather_nd(aop, circ_yx_hw2)
        dewarp_img_1hw = tf.expand_dims(dewarp_img_hw, axis=0)
        return dewarp_img_1hw

    def encode(self, data_dict):
        """encode raw dataset samples into desirable format

        Args:
            data_dict: dictionary {
                'image':        raw sensor image,
                'elevation':    solar elevation angle
            }

        Returns:
            dictionary {
                'image':        rectangular dewarped AoP image,
                'raw_aop':      fisheye AoP image,
                'raw_dop':      fisheye DoLP image,
                'elevation':    solar elevation angle
            }
        """
        raw_aop, raw_dop = self.raw2polarize(data_dict['image'])
        rect_aop = self.dewarp(raw_aop)

        ret = data_dict.copy()
        ret['input_b1hw'] = rect_aop
        ret['raw_aop'] = raw_aop
        ret['raw_dop'] = raw_dop
        return ret


class RegressorDataPipeline:
    """ TF data pipeline for training the regression model """

    DEFAULT_PARAMS=o(
        batch_size=128,
        # root directory that contains all training tf records
        train_root='/mnt/alvin/downsample20x/train',
        # encoder parameters
        encoder=DewarpEncoder.DEFAULT_PARAMS,
        # number of parallel disk I/O reads
        parallel_read=8,
        # number of parallel calls for encoding computations
        parallel_call=16,
        # size of shuffle buffer
        shuffle_size=1000,
        # size of prefetch buffer
        prefetch_size=1000,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        self.encoder = DewarpEncoder(self.p.encoder)

    def _parse_single(self, example_proto):
        feature_desc = {
            'image_png': tf.io.FixedLenFeature([], tf.string),
            'azimuth': tf.io.FixedLenFeature([], tf.float32),
            'elevation': tf.io.FixedLenFeature([], tf.float32),
            'dive': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_proto, feature_desc)
        features['image'] = tf.image.decode_image(features.pop('image_png'))
        features['image'].set_shape((None, None, None))

        return features

    def _process_sequence(self, example_proto):
        data_dict = self._parse_single(example_proto)
        data_dict = self.encoder.encode(data_dict)
        return (data_dict['input_b1hw'], data_dict['elevation'])

    def construct_train_dataset(self):
        file_pattern = os.path.join(self.p.train_root, '*.tfrecord')
        files = tf.io.matching_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files,
                                          num_parallel_reads=self.p.parallel_read)
        dataset = dataset.shuffle(self.p.shuffle_size)
        dataset = dataset.map(self._process_sequence,
                              num_parallel_calls=self.p.parallel_call)
        dataset = dataset.prefetch(self.p.prefetch_size)
        dataset = dataset.batch(self.p.batch_size)
        return dataset

    def construct_raw_dataset(self):
        """ construct raw dataset without any encoding
        for debug and visualization purposes"""
        file_pattern = os.path.join(self.p.train_root, '*.tfrecord')
        files = tf.io.matching_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files,
                                          num_parallel_reads=self.p.parallel_read)
        dataset = dataset.map(self._parse_single)
        return dataset

