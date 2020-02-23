import math
import os

import tensorflow as tf

from utils.params import ParamDict as o


class RegressorDataPipeline:
    """ TF data pipeline for training the regression model """

    DEFAULT_PARAMS=o(
        batch_size=128,
        # root directory that contains all training / validation tf records
        train_root='/mnt/alvin/downsample20x/train',
        val_root='/mnt/alvin/downsample20x/validation',
        # number of parallel disk I/O reads
        parallel_read=4,
        # size of shuffle buffer
        shuffle_size=1000,
        # size of prefetch buffer
        prefetch_size=1000,
    )

    def __init__(self, params=DEFAULT_PARAMS):
        self.p = params
        self.train_dataset = self.construct_dataset(self.p.train_root)
        self.val_dataset = self.construct_dataset(self.p.val_root)

    def _parse_single(self, example_proto):
        feature_desc = {
            'image_png': tf.io.FixedLenFeature([], tf.string),
            'azimuth': tf.io.FixedLenFeature([], tf.float32),
            'elevation': tf.io.FixedLenFeature([], tf.float32),
            'dive': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_proto, feature_desc)
        image = tf.image.decode_image(features.pop('image_png'))
        image.set_shape((None, None, None))
        image = tf.transpose(image, (2, 0, 1))

        return (image, features['elevation'])

    def construct_dataset(self, data_root):
        file_pattern = os.path.join(data_root, '*.tfrecord')
        files = tf.io.matching_files(file_pattern)
        dataset = tf.data.TFRecordDataset(files,
                                          num_parallel_reads=self.p.parallel_read)
        dataset = dataset.shuffle(self.p.shuffle_size)
        dataset = dataset.map(self._parse_single,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(self.p.prefetch_size)
        dataset = dataset.batch(self.p.batch_size)
        return dataset

