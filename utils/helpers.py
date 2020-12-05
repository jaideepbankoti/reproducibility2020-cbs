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

# Loads and preprocesses the data
def data_load(batches, data_fn):
	# getting the dataset
	ds_train, ds_test, ds_info = data_fn()
	# trainig pipeline
	ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	# ds_train = ds_train.cache()
	ds_train = ds_train.shuffle(1000)
	ds_train = ds_train.batch(batches)
	# ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

	# validation pipeline
	ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
	ds_test = ds_test.batch(batches)
	# ds_test = ds_test.cache()
	# ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

	return ds_train, ds_test

# helper functions for intial epoch
def init_epoch(ckpoint_path):
	return int(ckpoint_path.split('.')[1])

# a simple 2d gaussian filter, the shape of the filter is also a hyperparameter 
def gaussian_filter(shape =(3, 3), sigma=1):
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
	# print(sigma)
	kfilter = get_kernel_filter(inputs.shape[-1], sigma)
	kfilter = tf.convert_to_tensor(np.float32(kfilter))
	kernel = tf.Variable(
        initial_value=kfilter,
        trainable=False)
	# out = K.depthwise_conv2d(tf.cast(inputs, tf.float64), kfilter, padding='same')
	out = K.depthwise_conv2d(inputs, kfilter, padding='same')

	# out = tf.cast(out, tf.float32)
	return out

def scheduler(epoch, lr):
	if epoch != 0 and epoch % 10 == 0:
		return lr * tf.math.exp(-0.1)
	else:
		return lr