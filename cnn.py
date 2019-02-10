#importing libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing cnn
classifier = Sequential()

#1st layer: Convolutional layer
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3), activation = 'relu'))

#2nd layer: pooling layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

#3rd layer: flattening
classifier.add(Flatten())

#4th layer: full connection
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 1, activation='sigmoid'))

#compiliation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#fitting images into cnn
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set1 = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

testing_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set1,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=testing_set,
        validation_steps=2000)