import argparse
import glob
import logging
import random
import os

import pysolar.solar as sl
import tensorflow as tf

from tqdm import tqdm

from data.parser_v1 import FrameGroup, DiveMeta
from utils.tf.data import encode_string, encode_float, encode_image

class TFRecordGenerator:

    def __init__(self):
        self.args = self.parse_args()
        self.meta = DiveMeta(self.args.meta)
        self.files = glob.glob(
            os.path.join(self.args.input, '**/*.h5'), recursive=True)
        random.shuffle(self.files)

    def parse_args(self):
        parser = argparse.ArgumentParser() # TODO(alvin): add document
        parser.add_argument('-o', '--output', required=True, type=str,
                            help='directory to store the output tfrecord dataset')
        parser.add_argument('-i', '--input', required=True, type=str,
                            help='top level directory that contains all the h5 data')
        parser.add_argument('-m', '--meta', required=True, type=str,
                            help='path to the meta file that contains dive info')
        parser.add_argument('--shard-size', type=int, default=500,
                            help='maximum number of training samples per shard')
        return parser.parse_args()

    def shard_path(self, shard_id):
        fname = 'shard-{}.tfrecord'.format(shard_id)
        return os.path.join(self.args.output, fname)

    def serialize_sample(self, sample_dict):
        azimuth, elevation = sl.get_position(
            sample_dict['gps'][0], sample_dict['gps'][1], sample_dict['datetime'])
        feature = {
            'image_png':    encode_image(sample_dict['image']),
            'azimuth':      encode_float(azimuth),
            'elevation':    encode_float(elevation),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def run(self):
        shard_id = 0
        shard_cnt = 0
        writer = tf.io.TFRecordWriter(self.shard_path(shard_id))
        for h5 in tqdm(self.files):
            try:
                fgroup = FrameGroup(h5, self.meta)
            except (AssertionError, KeyError, OSError) as e:
                if type(e) == AssertionError:
                    logging.warn('Unmatched time: {}'.format(h5))
                else:
                    logging.warn('Corrupted h5: {}'.format(h5))
                continue
            shard_cnt += 1
            rand_idx = random.randint(0, fgroup.num_frames - 1)
            writer.write(self.serialize_sample(fgroup[rand_idx]))
            if shard_cnt >= self.args.shard_size:
                shard_cnt = 0
                shard_id += 1
                writer.close()
                writer = tf.io.TFRecordWriter(self.shard_path(shard_id))
        writer.close()

if __name__ == '__main__':
    TFRecordGenerator().run()

