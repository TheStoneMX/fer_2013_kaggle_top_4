

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import emotion_config as config
from pyimagesearch.preprocessing import ImageToArrayPreprocessor
from pyimagesearch.callbacks import EpochCheckpoint
from pyimagesearch.callbacks import TrainingMonitor
from pyimagesearch.io import HDF5DatasetGenerator
from pyimagesearch.nn.conv import EmotionVGGNet


from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import LearningRateScheduler

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.regularizers import l2

import tensorflow as TF

import numpy as np
import keras.backend as K
import argparse
import os


# USAGE
# python train_recognizer.py --checkpoints fer2013/checkpoints
# python train_recognizer.py --checkpoints fer2013/checkpoints --model fer2013/checkpoints/epoch_20.hdf5 --start-epoch 180

# keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
# keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
# keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
# keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# parameters
batch_size = 128
num_epochs = 200
input_shape = (48, 48, 1)
verbose = 1
num_classes = 7
patience = 25
base_path = 'models/'
l2_regularization=0.01

# def step_decay(epoch):
# 	# initialize the base initial learning rate, drop factor, and
# 	# epochs to drop every
# 	initAlpha = 0.01
# 	factor = 0.25
# 	dropEvery = 5

# 	# compute learning rate for the current epoch
# 	alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

# 	# return the learning rate
# 	return float(alpha)
# LearningRateScheduler(step_decay)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints",type=str, default='../datasets/fer2013/checkpoints', help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

# construct the training and testing image generators for data
# augmentation, then initialize the image preprocessor
trainAug = ImageDataGenerator(	rotation_range=10, 
								zoom_range=0.1, 
								horizontal_flip=True, 
								rescale=1 / 255.0, 
								fill_mode="nearest")


valAug = ImageDataGenerator(rescale=1 / 255.0)
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, aug=trainAug, preprocessors=[iap], classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, aug=valAug, preprocessors=[iap], classes=config.NUM_CLASSES)

# if there is no specific model checkpoint supplied, then initialize
# the network and compile the model
if args["model"] is None:
	print("[INFO] compiling model...")
	model = EmotionVGGNet.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
	# opt = Adam(lr=1e-3)
	# opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
	opt = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0)
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# otherwise, load the checkpoint from disk
else:
	print("[INFO] loading {}...".format(args["model"]))
	model = load_model(args["model"])

	# update the learning rate
	print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
	K.set_value(model.optimizer.lr, 1e-5)
	print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# construct the set of callbacks
figPath = os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.png"])
jsonPath = os.path.sep.join([config.OUTPUT_PATH,"vggnet_emotion.json"])

# # callbacks
# log_file_path = config.BASE_PATH + '_emotion_training.log'
# csv_logger = CSVLogger(log_file_path, append=False)

early_stop = EarlyStopping('val_acc', patience=patience)
reduce_lr = ReduceLROnPlateau('val_acc', factor=0.01, patience=20, verbose=1)
# reduce_lr = ReduceLROnPlateau(patience=9, verbose=1)

trained_models_path = config.BASE_PATH + '_best_model'

############################################################################
model_names = trained_models_path + '_epoch_{epoch:02d}_val_acc_{val_acc:.4f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names, 'val_acc', verbose=1, save_best_only=True)
############################################################################

# tensor_board = TF.tensorboard(log_dir=config.BASE_PATH, histogram_freq=2000, write_graph=True, write_images=False)

callbacks = [
	#EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
	model_checkpoint, early_stop, reduce_lr, 
	TrainingMonitor(figPath, startAt=args["start_epoch"])]

# callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr ]
# callbacks = [model_checkpoint, TrainingMonitor(figPath, jsonPath=jsonPath, startAt=args["start_epoch"])]]

# train the network
model.fit_generator(
	trainGen.generator(),
	steps_per_epoch=trainGen.numImages // config.BATCH_SIZE,
	validation_data=valGen.generator(),
	validation_steps=valGen.numImages // config.BATCH_SIZE,
	epochs=200,
	max_queue_size=10,
	callbacks=callbacks, verbose=1)

# close the databases
trainGen.close()
valGen.close()