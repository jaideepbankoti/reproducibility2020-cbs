"""
References:
	https://towardsdatascience.com/resuming-a-training-process-with-keras-3e93152ee11a
	https://www.pyimagesearch.com/2019/09/23/keras-starting-stopping-and-resuming-training/
  https://keras.io/api/callbacks/reduce_lr_on_plateau/
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from models.vgg16 import  VGG16Custom, train_step, test_step
from utils.dataload import cifar10_dataset, cifar100_dataset, svhn_dataset
import matplotlib.pyplot as plt
from os.path import isfile
from utils.helpers import set_seed, normalize_img, init_epoch, data_load
from tqdm import tqdm


# setting the seed -- [5, 10, 15, 20, 25]
set_seed(15)


input_shape = (32, 32, 3)
n_classes = 100
batches = 256


ds_train, ds_test = data_load(batches, data_fn=cifar100_dataset)
model = VGG16Custom(input_shape, n_classes)
# optim = tf.keras.optimizers.SGD(lr=0.1, decay=0.08, momentum=0.9)
optim    = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.5)
ckpt     = tf.train.Checkpoint(step=tf.Variable(1), mdoel=model, optimizer=optim)
manager  = tf.train.CheckpointManager(ckpt, directory='checkpoint/vgg16_checkpoint', max_to_keep=20)
emanager = tf.train.CheckpointManager(ckpt, directory='checkpoint/vgg16_echeckpoint', max_to_keep=20)
ckpt.restore(manager.latest_checkpoint)
START   = int(ckpt.step)//batches
if manager.latest_checkpoint:
	print('Restored from last checkpoint epoch : {0}'.format(START))
EPOCHS = 100
save_freq = 50

for epoch in range(START, EPOCHS):
	tloss = tf.keras.metrics.Mean()
	vloss = tf.keras.metrics.Mean()
	tacc  = tf.keras.metrics.SparseCategoricalAccuracy()
	vacc  = tf.keras.metrics.SparseCategoricalAccuracy()
	for train_itr, (X, Y) in enumerate(tqdm(ds_train), start=1):
		loss, Y_cap = train_step(model, optim, X, Y)
		tloss.update_state(loss)
		tacc.update_state(Y, Y_cap)
		ckpt.step.assign_add(1)
		if train_itr%save_freq == 0:
			manager.save()
		print('Train_itr: {0}\t Train_loss: {1}\t Train_acc: {2}'.format(train_itr,  loss, tacc.result()))

	for test_itr, (X, Y) in enumerate(tqdm(ds_test), start=1):
		loss, Y_cap = test_step(model, X, Y)
		vloss.update_state(loss)
		vacc.update_state(Y, Y_cap)
		print('Test_itr: {0}\t Test_loss: {1}\t Test_acc: {2}'.format(test_itr, loss, vacc.result()))

	with open('vgg16_log.txt', 'a') as file:
		file.write('Epoch: {0}\t Train_loss: {1}\t Train_acc: {2}\t Test_loss: {3}\t Test_acc: {4}\n'.format(epoch, tloss.result(), tacc.result(), vloss.result(), vacc.result()))

	ckpt.step.assign_add(1)
	emanager.save()
