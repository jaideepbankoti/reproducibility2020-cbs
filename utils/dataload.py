"""
References:
	https://www.tensorflow.org/datasets/splits
	https://www.gitmemory.com/issue/tensorflow/datasets/720/545541009
	https://www.tensorflow.org/datasets/keras_example
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# loading the mnist dataset
def mnist_dataset():
	(ds_train, ds_test), ds_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, as_supervised=True, data_dir='datasets', download=True, with_info=True)
	return ds_train, ds_test, ds_info

# loading the cifar10 dataset
def cifar10_dataset():
	(ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'], shuffle_files=True, as_supervised=True, data_dir='datasets', download=True, with_info=True)
	return ds_train, ds_test, ds_info

# loading the cifar100 dataset
def cifar100_dataset():
	(ds_train, ds_test), ds_info = tfds.load('cifar100', split=['train', 'test'], shuffle_files=True, as_supervised=True, data_dir='datasets', download=True, with_info=True)
	return ds_train, ds_test, ds_info

# loading the celeb_a dataset || can't seem to download this one
def celeb_a_dataset():
	(ds_train, ds_test), ds_info = tfds.load('celeb_a', split=['train', 'test'], shuffle_files=True, as_supervised=True, data_dir='datasets', download=True, with_info=True)
	return ds_train, ds_test, ds_info

# loading the svhn dataset
def svhn_dataset():
	(ds_train, ds_test), ds_info = tfds.load('svhn_cropped', split=['train', 'test'], shuffle_files=True, as_supervised=True, data_dir='datasets', download=True, with_info=True)
	return ds_train, ds_test, ds_info
