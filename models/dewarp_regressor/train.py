import argparse
import tensorflow as tf

from utils.params import ParamDict
from utils.tf.memory import enable_dynamic_gpu_memory

from models.dewarp_regressor.model import RegressorModel
from models.dewarp_regressor.data import RegressorDataPipeline

K = tf.keras

class Trainer:

    def __init__(self):
        K.backend.set_image_data_format('channels_first')
        enable_dynamic_gpu_memory()
        self.args = self.parse_arguments()
        self.p = ParamDict.from_file(self.args.params)
        self.model = RegressorModel(self.p.regressor).create_model()
        self.dataset = RegressorDataPipeline(self.p.data).construct_train_dataset()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--params', required=True, type=str,
                            help='path to the params file')
        parser.add_argument('--log-dir', required=True, type=str,
                            help='directory to store the logged checkpoints')
        return parser.parse_args()

    def run(self):
        optimizer = self.p.trainer.optimizer_func()
        self.model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model.fit(self.dataset)

if __name__ == '__main__':
    Trainer().run()
