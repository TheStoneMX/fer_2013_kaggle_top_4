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
		model.add(Conv2D(32, # number of filters
						(3, 3), # filter or kernel
						padding="same", #  same padding to ensure the size of output of the convolution operation matches the input 
						kernel_initializer="he_normal", 
						input_shape=inputShape))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal", padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Block #2: second CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",padding="same"))
		model.add(ELU()) 
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal",padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# Block #3: third CONV => RELU => CONV => RELU => POOL
		# layer set
		model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal",padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))

		model.add(Conv2D(256, (3, 3), kernel_initializer="he_normal",padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		model.add(Conv2D(512, (3, 3), kernel_initializer="he_normal",padding="same"))
		model.add(ELU())
		model.add(BatchNormalization(axis=chanDim))

		# Block #4: first set of FC => RELU layers
		model.add(Flatten()) # n order to apply our fully-connected layer, we ﬁrst need to ﬂatten 
		model.add(Dense(64, kernel_initializer="he_normal"))
		model.add(ELU())
		model.add(BatchNormalization())
		model.add(Dropout(0.25))

		# Block #6: second set of FC => RELU layers
		model.add(Dense(64, kernel_initializer="he_normal"))
		model.add(ELU())
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# Block #7: softmax classifier
		model.add(Dense(classes, kernel_initializer="he_normal"))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model

if __name__ == "__main__":
	# visualize the network architecture
	from keras.utils import plot_model
	from keras.regularizers import l2
	model = EmotionVGGNet.build(48, 48, 1, 6)
	plot_model(model, to_file="model.png", show_shapes=True,
		show_layer_names=True)