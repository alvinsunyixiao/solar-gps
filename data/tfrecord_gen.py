import argparse
import glob
import random
import os

import numpy as np
import pysolar.solar as sl
import tensorflow as tf

from multiprocessing import Process
from tqdm import tqdm

from data.parser_v1 import FrameGroup, DiveMeta
from utils.console import print_warn, print_ok
from utils.params import ParamDict
from utils.tf.data import encode_string, encode_float, encode_image

class TFRecordGenerator:

    DEFAULT_PARAMS=ParamDict(
        # root directory of the raw h5 data
        data_root='/mnt/data',
        # dives used for training
        train=['010720DIVE1',
               '010820DIVE1', '010820DIVE2', '010820DIVE3',
               '010920DIVE1', '010920DIVE2', '010920DIVE3',
               '011020DIVE1', '011020DIVE2'],
        # dives used for validation
        validation=['011020DIVE3'],
        # meta file that store dive information
        meta='/home/alvin_sun/01xx20_dive_data.txt',
        # maximum number of samples per tfrecord shard
        shard_size=1000,
        # randomly sample this portion of the original dive data per h5 file
        downsample=0.05,
    )

    def __init__(self):
        self.args = self.parse_args()
        self.p = self.DEFAULT_PARAMS
        if self.args.params is not None:
            self.p = ParamDict.from_file(self.args.params)
        self.meta = DiveMeta(self.p.meta)
        self.train = self._scan_and_sample_dataset(self.p.train)
        self.validation = self._scan_and_sample_dataset(self.p.validation)
        print_ok('Successfully loaded {} training examples'.format(len(self.train)))
        print_ok('Successfully loaded {} validation examples'.format(len(self.validation)))
        # make sure output directories are setup
        os.makedirs(os.path.join(self.args.output, 'train'), exist_ok=True)
        os.makedirs(os.path.join(self.args.output, 'validation'), exist_ok=True)

    def _scan_and_sample_dataset(self, dives):
        roots = [os.path.join(self.p.data_root, n) for n in dives]
        ret = []
        for root in roots:
            h5_files = glob.glob(os.path.join(root, '*.h5'))
            for h5 in h5_files:
                try:
                    fgroup = FrameGroup(h5, self.meta)
                except (AssertionError, KeyError, OSError) as e:
                    if type(e) == AssertionError:
                        print_warn('Unmatched time: {}'.format(h5))
                    else:
                        print_warn('Corrupted h5: {}'.format(h5))
                    continue
                num_samples = int(self.p.downsample * fgroup.num_frames)
                indices = np.random.choice(
                    fgroup.num_frames, size=num_samples, replace=False)
                ret.extend([(h5, int(idx)) for idx in indices])
        random.shuffle(ret)
        return ret

    def parse_args(self):
        parser = argparse.ArgumentParser() # TODO(alvin): add document
        parser.add_argument('-o', '--output', required=True, type=str,
                            help='directory to store the output tfrecord dataset')
        parser.add_argument('-p', '--params', type=str, default=None,
                            help='(optional) path to a params file')
        parser.add_argument('-j', '--jobs', type=int, default=12,
                            help='number of parallel workers [default to 12]')
        return parser.parse_args()

    def shard_path(self, shard_id, training=True):
        sub_dir = 'train' if training else 'validation'
        fname = 'shard-{}.tfrecord'.format(shard_id)
        return os.path.join(self.args.output, sub_dir, fname)

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

    def worker(self, worker_id, training=True):
        example_list = self.train if training else self.validation
        num_shards = int(np.ceil(len(example_list) / self.p.shard_size))
        for shard_id in range(num_shards):
            if shard_id % self.args.jobs != worker_id:
                continue
            with tf.io.TFRecordWriter(self.shard_path(shard_id, training)) as writer:
                examples = example_list[shard_id * self.p.shard_size: \
                                        (shard_id+1) * self.p.shard_size]
                examples = tqdm(examples, leave=False, position=worker_id,
                                desc='Worker {} on shard {}'.format(worker_id, shard_id))
                for h5_file, idx in examples:
                    fgroup = FrameGroup(h5_file, self.meta)
                    writer.write(self.serialize_sample(fgroup[idx]))

    def run(self):
        pool = []
        for i in range(self.args.jobs):
            p = Process(target=self.worker, args=(i, True))
            p.start()
            pool.append(p)
        # wait for all workers to finish
        while len(pool):
            pool.insert(0, pool.pop())
            if not pool[-1].is_alive():
                pool[-1].join()
                del pool[-1]


if __name__ == '__main__':
    tqdm.get_lock()
    TFRecordGenerator().run()

