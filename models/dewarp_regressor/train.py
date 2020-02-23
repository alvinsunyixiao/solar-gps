import argparse
import os
import time
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
        self.data = RegressorDataPipeline(self.p.data)

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
        log_dir = os.path.join(self.args.log_dir, time.strftime('%Y-%m-%d-%H-%M-%S'))
        os.makedirs(log_dir)
        callbacks = [
            K.callbacks.ModelCheckpoint(
                filepath=os.path.join(log_dir, '{epoch:03d}.h5'),
                save_weights_only=True,
            ),
            K.callbacks.TensorBoard(
                log_dir=log_dir,
                update_freq=self.p.trainer.log_steps,
                histogram_freq=1,
               profile_batch=0,
            ),
            K.callbacks.LearningRateScheduler(self.p.trainer.lr_schedule),
        ]
        self.model.fit(self.data.train_dataset,
                       validation_data=self.data.val_dataset,
                       epochs=self.p.trainer.num_epochs,
                       callbacks=callbacks)

if __name__ == '__main__':
    Trainer().run()
