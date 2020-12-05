"""
This module contains all the necessary helper functions
https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686
https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/noise.py#L32-L82
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import keras_export
from random import seed as rand_seed

def set_seed(seed):
	rand_seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

# Normalizing images, shall be used in map function
def normalize_img(image, label):
	return tf.cast(image, tf.float32) / 255., label

# helper functions for intial epoch
def init_epoch(ckpoint_path):
	return int(ckpoint_path.split('.')[1])

# a simple 2d gaussian filter, the shape of the filter is also a hyperparameter 
def gaussian_filter(shape =(3,3), sigma=1):
    x, y = shape[0] // 2, shape[1] // 2
    gaussian_grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
    gaussian_filter = np.exp(-gaussian_grid)/(2*np.pi*sigma**2)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

# defining a custom gaussian_kernel layer
def get_kernel_filter(input_shape, stddev):
	kfilter = gaussian_filter(sigma=stddev)
	kfilter = np.expand_dims(kfilter, axis=-1)
	kfilter = np.repeat(kfilter, input_shape, axis=-1)
	kfilter = np.expand_dims(kfilter, axis=-1)
	return kfilter

# defining a custom gaussian kernel function
def gaussian_kernel_layer(inputs, kfilter):
	kernel = tf.Variable(
        initial_value=kfilter,
        trainable=False, dtype=tf.float64)
	out = K.depthwise_conv2d(tf.cast(inputs, tf.float64), kfilter, padding='same')
	return out"""
This module contains all the necessary helper functions
https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python/17201686#17201686
https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/layers/noise.py#L32-L82
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.util.tf_export import keras_export
from random import seed as rand_seed

def set_seed(seed):
	rand_seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)

# Normalizing images, shall be used in map function
def normalize_img(image, label):
	return tf.cast(image, tf.float32) / 255., label

# helper functions for intial epoch
def init_epoch(ckpoint_path):
	return int(ckpoint_path.split('.')[1])

# a simple 2d gaussian filter, the shape of the filter is also a hyperparameter 
def gaussian_filter(shape =(3,3), sigma=1):
    x, y = shape[0] // 2, shape[1] // 2
    gaussian_grid = np.array([[((i**2+j**2)/(2.0*sigma**2)) for i in range(-x, x+1)] for j in range(-y, y+1)])
    gaussian_filter = np.exp(-gaussian_grid)/(2*np.pi*sigma**2)
    gaussian_filter /= np.sum(gaussian_filter)
    return gaussian_filter

# defining a custom gaussian_kernel layer
def get_kernel_filter(input_shape, stddev):
	kfilter = gaussian_filter(sigma=stddev)
	kfilter = np.expand_dims(kfilter, axis=-1)
	kfilter = np.repeat(kfilter, input_shape, axis=-1)
	kfilter = np.expand_dims(kfilter, axis=-1)
	return kfilter

# defining a custom gaussian kernel function
def gaussian_kernel_layer(inputs, sigma=1):
	kfilter = get_kernel_filter(inputs.shape[-1], sigma)
	kernel = tf.Variable(
        initial_value=kfilter,
        trainable=False, dtype=tf.float64)
	out = K.depthwise_conv2d(tf.cast(inputs, tf.float64), kfilter, padding='same')
	return out