import tensorflow as tf

def sometimes(input_bchw, prob, fn):
    mask = tf.less(tf.random.uniform([tf.shape(input_bchw)[0], 1, 1, 1]), prob)
    return tf.where(mask, x=fn(input_bchw), y=input_bchw)

def additive_gaussian_noise(input_bchw, loc, scale):
    std = tf.random.uniform([], scale[0], scale[1])
    noise = tf.random.normal(tf.shape(input_bchw), mean=loc, stddev=std)
    return tf.clip_by_value(input_bchw + noise, 0, 255)
