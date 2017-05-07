# Convolutional Neural Network

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import *

classifier = Sequential()
classifier.add(Conv2D(32, 3, strides=(3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D())
#classifier.add(Conv2D(64, 3, strides=(3,3), activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(128, activation='relu'))
#since we have binary output, we use sigmoid, else we would have used softmax as the activation function
classifier.add(Dense(1, activation='sigmoid'))

sgd= SGD(lr=10e-5,momentum=0.99, decay=0.9999, nesterov=True)

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the CNN to the images
# Image augmentation to avoid overfitting
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/train', target_size=(64, 64), batch_size=32, class_mode='binary')

validation_generator = test_datagen.flow_from_directory( 'dataset/test', target_size=(64, 64), batch_size=32, class_mode='binary')

classifier.fit_generator( train_generator, steps_per_epoch=100, epochs=5, validation_data=validation_generator, validation_steps=40)


"""
# Single Prediction
import numpy as np
from keras.preprocessing import image

test_dog = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_dog = image.img_to_array(test_dog)
test_dog = np.expand_dims(test_dog, 0)
result = classifier.predict(test_dog)

prediction = "n/a"
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

test_cat = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))

test_cat = image.img_to_array(test_cat)
test_cat = np.expand_dims(test_cat, 0)
result = classifier.predict(test_cat)

prediction = "n/a"
if result[0][0] == 1:
    prediction = "dog"
else:
    prediction = "cat"

prediction
"""