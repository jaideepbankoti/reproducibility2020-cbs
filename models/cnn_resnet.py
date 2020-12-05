import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

initializer = tf.keras.initializers.HeNormal()
regularizer = tf.keras.regularizers.l2(1e-4)


# this resnet18 module can run cifar10, cifar100, svhn; printing the shapes for debugging
class ResNet18(Model):
	def __init__(self, input_shape, n_classes):
		super(ResNet18, self).__init__()
		input = keras.Input(shape=input_shape)
		
		self.conv_1 = layers.Conv2D(filters=64, name='c1', kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bn_1 = layers.BatchNormalization()

		# Basic block -1
		self.bb1_conv1 	= layers.Conv2D(filters=64,name='c2', kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb1_act1 	= layers.Activation('relu')
		self.bb1_bn1 	= layers.BatchNormalization(name='bb1_bn1')
		
		self.bb1_conv2 	= layers.Conv2D(filters=64,name='c3', kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb1_act2 	= layers.Activation('relu')
		self.bb1_bn2 	= layers.BatchNormalization(name='bb1_bn2')

		self.bb1_add 	= layers.Add(name='bb1_add')
		self.bb1_act 	= layers.Activation('relu')

		# self.bb1_conv3 		= layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		# self.bb1_act3 		= layers.Activation('relu')
		# self.bb1_bn3 		= layers.BatchNormalization()

		# Basic block -2
		self.bb2_conv1 = layers.Conv2D(filters=64, name='c4', kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb2_act1 = layers.Activation('relu', name='bb2_act1')
		self.bb2_bn1 = layers.BatchNormalization(name='bb2bn1')
		
		self.bb2_conv2 = layers.Conv2D(filters=64,name='c5', kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb2_act2 = layers.Activation('relu', name='bb2_act2')
		self.bb2_bn2 = layers.BatchNormalization(name='bb2_bn2')

		self.bb2_add = layers.Add(name='bb2_add')
		self.bb2_act = layers.Activation('relu', name='bb2_act')

		# Basic block -3
		self.bb3_conv1 = layers.Conv2D(filters=128, name='c6', kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb3_act1 = layers.Activation('relu', name='bb3_act1')
		self.bb3_bn1 = layers.BatchNormalization(name='bb2bn1')
		
		self.bb3_conv2 = layers.Conv2D(filters=128, name='c7',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb3_act2 = layers.Activation('relu', name='bb3_act2')
		self.bb3_bn2 = layers.BatchNormalization(name='bb3_bn2')

		self.bb3_add = layers.Add(name='bb3_add')
		self.bb3_act = layers.Activation('relu', name='bb3_act')

		self.bb3_conv3 		= layers.Conv2D(filters=128, name='c8',kernel_size=(1, 1), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb3_act3 		= layers.Activation('relu', name='bb3_act')
		self.bb3_bn3 		= layers.BatchNormalization(name='bb3_bn3')

		# Basic block -4
		self.bb4_conv1 = layers.Conv2D(filters=128, name='c9',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb4_act1 = layers.Activation('relu', name='bb4_act1')
		self.bb4_bn1 = layers.BatchNormalization(name='bb4_bn1')
		
		self.bb4_conv2 = layers.Conv2D(filters=128, name='c10',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb4_act2 = layers.Activation('relu', name='bb4_act2')
		self.bb4_bn2 = layers.BatchNormalization(name='bb4_bn2')

		self.bb4_add = layers.Add(name='bb4_add')
		self.bb4_act = layers.Activation('relu', name='bb4_act')

		# Basic block -5
		self.bb5_conv1 = layers.Conv2D(filters=256, name='c11',kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb5_act1 = layers.Activation('relu', name='bb5_act1')
		self.bb5_bn1 = layers.BatchNormalization(name='bb5_bn1')
		
		self.bb5_conv2 = layers.Conv2D(filters=256, name='c12',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb5_act2 = layers.Activation('relu', name='bb5_act2')
		self.bb5_bn2 = layers.BatchNormalization(name='bb5_bn2')

		self.bb5_add = layers.Add(name='bb5_add')
		self.bb5_act = layers.Activation('relu', name='bb5_act')

		self.bb5_conv3 		= layers.Conv2D(filters=256, name='c13',kernel_size=(1, 1), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb5_act3 		= layers.Activation('relu', name='bb5_act3')
		self.bb5_bn3 		= layers.BatchNormalization(name='bb5_bn3')

		# Basic block -6
		self.bb6_conv1 = layers.Conv2D(filters=256, name='c14',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb6_act1 = layers.Activation('relu', name='bb6_act1')
		self.bb6_bn1 = layers.BatchNormalization(name='bb6_bn1')
		
		self.bb6_conv2 = layers.Conv2D(filters=256, name='c15',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb6_act2 = layers.Activation('relu', name='bb6_act2')
		self.bb6_bn2 = layers.BatchNormalization(name='bb6_bn2')

		self.bb6_add = layers.Add(name='bb6_add')
		self.bb6_act = layers.Activation('relu', name='bb6_act')

		# Basic block -7
		self.bb7_conv1 = layers.Conv2D(filters=512, name='c16',kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb7_act1 = layers.Activation('relu', name='bb7_act1')
		self.bb7_bn1 = layers.BatchNormalization(name='bb7_bn1')
		
		self.bb7_conv2 = layers.Conv2D(filters=512, name='c17',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb7_act2 = layers.Activation('relu', name='bb7_act2')
		self.bb7_bn2 = layers.BatchNormalization(name='bb7_bn2')

		self.bb7_add = layers.Add(name='bb7_add')
		self.bb7_act = layers.Activation('relu', name='bb7_act')

		self.bb7_conv3 		= layers.Conv2D(filters=512, name='c18',kernel_size=(1, 1), strides=(2, 2), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb7_act3 		= layers.Activation('relu', name='bb7_act3')
		self.bb7_bn3 		= layers.BatchNormalization(name="bb7_bn3")

		# Basic block -8
		self.bb8_conv1 = layers.Conv2D(filters=512, name='c20',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb8_act1 = layers.Activation('relu', name='bb8_act1')
		self.bb8_bn1 = layers.BatchNormalization(name='bb8_bn1')
		
		self.bb8_conv2 = layers.Conv2D(filters=512, name='c21',kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer=initializer, kernel_regularizer=regularizer)
		self.bb8_act2 = layers.Activation('relu', name='bb8_act')
		self.bb8_bn2 = layers.BatchNormalization(name='bb8_bn2')

		self.bb8_add = layers.Add(name='bb8_add')
		self.bb8_act = layers.Activation('relu', name='bb8_act')



		self.avg_pool_5 = layers.AveragePooling2D(pool_size=(4, 4), name='avg_pool_5')
		self.flat = layers.Flatten()
		self.fc_6 = layers.Dense(n_classes, kernel_initializer=initializer, kernel_regularizer=regularizer, name='fc_6')
		self.out = layers.Activation('softmax', name='softmax')

	def call(self, x):
		x = self.bn_1(self.conv_1(x))

		res = self.bb1_conv1(x)
		res = self.bb1_bn1(self.bb1_act1(res))
		res = self.bb1_conv2(res)
		res = self.bb1_bn2(self.bb1_act2(res))
		# ski = self.bb1_conv3(x)
		res = self.bb1_add([self.bb1_bn2(self.bb1_act2(res)), x])
		x = self.bb1_act(res)

		res = self.bb2_conv1(x)
		res = self.bb2_bn1(self.bb2_act1(res))
		res = self.bb2_conv2(res)
		res = self.bb2_bn2(self.bb2_act2(res))
		res = self.bb2_add([self.bb2_bn2(self.bb2_act2(res)), x])
		x = self.bb2_act(res)

		res = self.bb3_conv1(x)
		res = self.bb3_bn1(self.bb3_act1(res))
		res = self.bb3_conv2(res)
		res = self.bb3_bn2(self.bb3_act2(res))
		ski = self.bb3_conv3(x)
		res = self.bb3_add([self.bb3_bn2(self.bb3_act2(res)), self.bb3_bn3(self.bb3_act3(ski))])
		x = self.bb3_act(res)

		res = self.bb4_conv1(x)
		res = self.bb4_bn1(self.bb4_act1(res))
		res = self.bb4_conv2(res)
		res = self.bb4_bn2(self.bb4_act2(res))
		res = self.bb4_add([self.bb4_bn2(self.bb4_act2(res)), x])
		x = self.bb4_act(res)

		res = self.bb5_conv1(x)
		res = self.bb5_bn1(self.bb5_act1(res))
		res = self.bb5_conv2(res)
		res = self.bb5_bn2(self.bb5_act2(res))
		ski = self.bb5_conv3(x)
		res = self.bb5_add([self.bb5_bn2(self.bb5_act2(res)), self.bb5_bn3(self.bb5_act3(ski))])
		x = self.bb5_act(res)

		res = self.bb6_conv1(x)
		res = self.bb6_bn1(self.bb6_act1(res))
		res = self.bb6_conv2(res)
		res = self.bb6_bn2(self.bb6_act2(res))
		res = self.bb6_add([self.bb6_bn2(self.bb6_act2(res)), x])
		x = self.bb6_act(res)

		res = self.bb7_conv1(x)
		res = self.bb7_bn1(self.bb7_act1(res))
		res = self.bb7_conv2(res)
		res = self.bb7_bn2(self.bb7_act2(res))
		ski = self.bb7_conv3(x)
		res = self.bb7_add([self.bb7_bn2(self.bb7_act2(res)), self.bb7_bn3(self.bb7_act3(ski))])
		x = self.bb7_act(res)

		res = self.bb8_conv1(x)
		res = self.bb8_bn1(self.bb8_act1(res))
		res = self.bb8_conv2(res)
		res = self.bb8_bn2(self.bb8_act2(res))
		res = self.bb8_add([self.bb8_bn2(self.bb8_act2(res)), x])
		x = self.bb8_act(res)
		
		x = self.avg_pool_5(x)
		x = self.flat(x)
		x = self.fc_6(x)
		x = self.out(x)
		return x

def train_step(model, optim, X, Y):
	with tf.GradientTape() as tape:
		Y_cap = model(X, training=True)
		loss  = losses.SparseCategoricalCrossentropy()(Y, Y_cap)
	variables = model.trainable_variables
	gradeints = tape.gradient(loss, variables)
	optim.apply_gradients(zip(gradeints, variables))
	return loss, Y_cap

def test_step(model, X, Y):
	Y_cap = model(X, training=False)
	loss  = losses.SparseCategoricalCrossentropy()(Y, Y_cap)
	return loss, Y_cap
