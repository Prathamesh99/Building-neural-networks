#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing cnn
classifier = Sequential()

#1st layer: Convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(32,32,3), activation = 'relu'))

#2nd layer: pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#3rd layer: flattening
classifier.add(Flatten())

#4th layer: full connection
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))
