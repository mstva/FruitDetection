#(1) Import packages
import os
import numpy as np

from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


import matplotlib.pyplot as plt
%matplotlib inline

#(2) Preprocessing the data
train_path = '../input/fruits/fruits-360_dataset/fruits-360/Training'
test_path = '../input/fruits/fruits-360_dataset/fruits-360/Test'

train_labels = os.listdir(train_path)
test_labels = os.listdir(test_path)

train_generator = ImageDataGenerator(rescale=1./255)
test_generator = ImageDataGenerator(rescale=1./255)

train_data = train_generator.flow_from_directory(train_path, batch_size=32, classes=train_labels, target_size=(64,64))
test_data = test_generator.flow_from_directory(test_path, batch_size=32, classes=train_labels, target_size=(64,64))

#(3) Building the Keras model
cnn = Sequential()

cnn.add(Conv2D(16, (3, 3), input_shape = (64, 64, 3), padding = "same", activation = "relu"))
cnn.add(MaxPooling2D())

cnn.add(Conv2D(32, (3,3), padding='same', activation='relu'))
cnn.add(MaxPooling2D())

cnn.add(Conv2D(64, (3,3), padding='same', activation='relu'))
cnn.add(MaxPooling2D())

cnn.add(Flatten())

cnn.add(Dropout(0.25))

cnn.add(Dense(256, activation = "relu"))
cnn.add(Dense(len(train_labels), activation = "softmax"))


#(4) Compiling the model
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

#(5) Training the model
history_cnn = cnn.fit(train_data, steps_per_epoch=1000, epochs=5, validation_steps=400, validation_data=test_data)

#(6) Plotting the accuracy and validation accuracy
plt.plot(history_cnn.history['accuracy'])
plt.plot(history_cnn.history['val_accuracy'])

#(7) Evaluation of the model
score = cnn.evaluate(test_data)

#(8) Saving the model
cnn.save("cnn_model.h5")

#(9) Loading the model
new_cnn = load_model("cnn_model.h5")