import tensorflow as tf

from models.dewarp_regressor.data import RegressorDataPipeline, DewarpEncoder
from models.dewarp_regressor.model import RegressorModel

from utils.params import ParamDict as o

regressor = RegressorModel.DEFAULT_PARAMS
data = RegressorDataPipeline.DEFAULT_PARAMS

shared = o(
    input_hw=(32, 512),
    batch_size=128,
)

trainer = o(
    optimizer_func=lambda: tf.keras.optimizers.SGD(1e-2, 0.9),
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
        params.data.encoder,
        params.regressor,
        params.trainer,
    )
    for p in require_shared:
        p.update(params.shared)

resolve_dependancies(PARAMS)
