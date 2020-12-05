"""
	References:
		https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
	https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from utils.helpers import gaussian_kernel_layer


class VGG16CustomCbs(Model):
	def __init__(self, input_shape, n_classes=10):
		super(VGG16CustomCbs, self).__init__()
		n_filters = [64, 128, 256, 512]
		input = keras.Input(shape=input_shape)
		initializer = tf.keras.initializers.HeNormal()
		regularizer = tf.keras.regularizers.l2(5e-4)	# from original paper  

		self.conv_1 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.act_1 = layers.Activation('relu')
		self.conv_2 = layers.Conv2D(filters=n_filters[0], kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.pool_2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
		self.act_2 = layers.Activation('relu')

		self.conv_3 = [layers.Conv2D(filters=n_filters[1], kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer) for i in range(2)]
		self.pool_3 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
		self.act_3 = layers.Activation('relu')

		self.conv_4 = [layers.Conv2D(filters=n_filters[2], kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer) for i in range(3)]
		self.pool_4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
		self.act_4 = layers.Activation('relu')

		self.conv_5 = [layers.Conv2D(filters=n_filters[3], kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer) for i in range(3)]
		self.pool_5 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
		self.act_5 = layers.Activation('relu')

		self.conv_6 = [layers.Conv2D(filters=n_filters[3], kernel_size=(3, 3), padding="same", kernel_initializer=initializer, kernel_regularizer=regularizer) for i in range(3)]
		self.pool_6 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
		self.act_6 = layers.Activation('relu')

		self.flat   = layers.Flatten()
		self.fc_7   = layers.Dense(units=4096, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.drop_7 = layers.Dropout(0.5)
		self.fc_8   = layers.Dense(units=4096, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.drop_8 = layers.Dropout(0.5)
		self.out    = layers.Dense(units=n_classes, activation="softmax", kernel_initializer=initializer, kernel_regularizer=regularizer)

	def call(self, x, sigma):
		x = self.conv_1(x)
		x = self.act_1(gaussian_kernel_layer(x, sigma))
		x = self.conv_2(x)
		x = gaussian_kernel_layer(x, sigma)
		x = self.act_2(self.pool_2(x))
		
		for i in range(2):
			x = self.conv_3[i](x)
			x = gaussian_kernel_layer(x, sigma)
		x = self.act_3(self.pool_3(x))
		
		for i in range(3):
			x = self.conv_4[i](x)
			x = gaussian_kernel_layer(x, sigma)
		x = self.act_4(self.pool_4(x))
		
		for i in range(3):
			x = self.conv_5[i](x)
			x = gaussian_kernel_layer(x, sigma)
		x = self.act_5(self.pool_5(x))

		for i in range(3):
			x = self.conv_6[i](x)
			x = gaussian_kernel_layer(x, sigma)
		x = self.act_6(self.pool_6(x))
		x = self.flat(x)
		x = self.fc_7(x)
		x = self.drop_7(x)
		x = self.fc_8(x)
		x = self.drop_8(x)
		x = self.out(x)
		return x

def train_step(model, optim, X, Y, sigma=1):
	with tf.GradientTape() as tape:
		Y_cap = model(X, sigma, training=True)
		loss  = losses.SparseCategoricalCrossentropy()(Y, Y_cap)
	variables = model.trainable_variables
	gradeints = tape.gradient(loss, variables)
	optim.apply_gradients(zip(gradeints, variables))
	return loss, Y_cap

def test_step(model, X, Y, sigma=1):
	Y_cap = model(X, sigma, training=False)
	loss  = losses.SparseCategoricalCrossentropy()(Y, Y_cap)
	return loss, Y_cap