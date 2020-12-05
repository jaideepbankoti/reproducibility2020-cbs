# https://androidkt.com/get-output-of-intermediate-layers-keras/
from helpers  import normalize_img, gaussian_filter
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
import cv2

image = cv2.imread('cat.jpg')

image = np.float32(image / 255.0)
# cv2.destroyAllWindows()

# defining a custom gaussian kernel function
def gaussian_kernel_layer(inputs, sigma=1):
	kfilter = get_kernel_filter(inputs.shape[-1], sigma)
	print('hey', kfilter.shape)
	kernel = tf.Variable(
        initial_value=kfilter,
        trainable=False, dtype=tf.float64)
	out = K.depthwise_conv2d(tf.cast(inputs, tf.float64), kfilter, padding='same')
	return out

def get_kernel_filter(input_shape, stddev):
	kfilter = gaussian_filter(sigma=stddev)
	kfilter = np.expand_dims(kfilter, axis=-1)
	kfilter = np.repeat(kfilter, input_shape, axis=-1)
	kfilter = np.expand_dims(kfilter, axis=-1)
	return kfilter

def test_model(input_shape, sigma=5):
	input = keras.Input(shape = input_shape)
	# x = gaussian_kernel(sigma, name='G')(input)
	kfilter = get_kernel_filter(3, sigma)
	x = gaussian_kernel_layer(input, sigma)
	model = Model(inputs=input, outputs=x)
	return model

def visualize_conv_layer(layer_name):
	layer_output=model.get_layer(name='G').output
	intermediate_model=tf.keras.models.Model(inputs=model.input,outputs=layer_output)
	intermediate_prediction=intermediate_model.predict(x_train[2].reshape(1,28,28,1))
	row_size=4
	col_size=8
	img_index=0
	print(np.shape(intermediate_prediction))
	fig,ax=plt.subplots(row_size,col_size,figsize=(10,8))
	for row in range(0,row_size):
		for col in range(0,col_size):
			ax[row][col].imshow(intermediate_prediction[0, :, :, img_index], cmap='gray')
			img_index=img_index+1


net = test_model(input_shape = (708,760,3))
image_tensor = K.constant(np.expand_dims(image, axis=0))
print(net.summary())
# print(net.get_layer(name='G').stddev)
# # print(net.get_output_at(0))
# im = net(image_tensor)
# im = tf.squeeze(im, axis=0)
# print(im.numpy().shape)
# # cv2.imshow('asfd', im.numpy())
# # cv2.waitKey(3000)
# print(net.get_layer(name='G').get_weights())

# # gaussian
# kfilter = get_kernel_filter(3, stddev=5)
# print(net.get_layer(name='G').set_weights(kfilter))
# print(kfilter.shape)

im = net(image_tensor)
im = tf.squeeze(im, axis=0)
print(im.numpy().shape)
cv2.imshow('asfd', im.numpy())
cv2.waitKey(3000)

# print(net.get_layer(name='G').get_weights())