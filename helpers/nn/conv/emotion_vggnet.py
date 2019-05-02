# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class EmotionVGGNet:
	@staticmethod
	def build(	width, # The width of the input images that will be used to train the network (i.e., number of columns in the matrix).  
				height, # The height of our input images (i.e., the number of rows in the matrix)
				depth, # The number of channels in the input image
				classes): # The total number of classes that our network should learn to predict. For Animals, classes=3 and for CIFAR-10, classes=10. 
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

    	# Block #1: first CONV => RELU => CONV => RELU => POOL
		# layer set
        model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:], name='block1_conv1'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

        model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))
        
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))

        model.add(Flatten())

        model.add(Dense(4096))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(4096, name='fc2'))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes))
        model.add(BatchNormalization()) if BATCH_NORM else None
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.0005, decay=0, nesterov=True)

        model.compile(loss='categorical_

		return model

if __name__ == "__main__":
	# visualize the network architecture
	from keras.utils import plot_model
	from keras.regularizers import l2
	model = EmotionVGGNet.build(48, 48, 1, 6)
	plot_model(model, to_file="model.png", show_shapes=True,
		show_layer_names=True)