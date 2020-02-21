import tensorflow as tf

def enable_dynamic_gpu_memory():
    """ enable dynamic gpu memory allocation for all visable gpus """
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

