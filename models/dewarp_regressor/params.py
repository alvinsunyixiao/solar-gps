import tensorflow as tf

from models.dewarp_regressor.data import RegressorDataPipeline
from models.dewarp_regressor.model import RegressorModel

from utils.params import ParamDict as o

regressor = RegressorModel.DEFAULT_PARAMS
data = RegressorDataPipeline.DEFAULT_PARAMS

def default_lr_schedule(epoch):
    if epoch < 40:
        return 1e-3
    elif epoch < 60:
        return 1e-4
    elif epoch < 80:
        return 1e-5
    else:
        return 1e-6

shared = o(
    input_hw=(2048, 2448),
    batch_size=128,
)

trainer = o(
    optimizer_func=lambda: tf.keras.optimizers.Adam(),
    num_epochs=100,
    log_steps=20,
    lr_schedule=default_lr_schedule,
)

PARAMS = o(
    data=RegressorDataPipeline.DEFAULT_PARAMS,
    regressor=RegressorModel.DEFAULT_PARAMS,
    shared=shared,
    trainer=trainer,
)

def resolve_dependancies(params):
    require_shared = (
        params.data,
        params.regressor,
        params.trainer,
    )
    for p in require_shared:
        p.update(params.shared)

resolve_dependancies(PARAMS)
