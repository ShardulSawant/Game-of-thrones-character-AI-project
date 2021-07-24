# Importing the libraries
from keras.layers import Activation, Dropout, Flatten, Dense,Conv2D, MaxPool2D,Input
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np



batch_size = 10
img_rows = 128
img_cols = 128

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=15,
    # shear_range=0.2,
    # zoom_range=0.2,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # # vertical_flip=True,
    # horizontal_flip=True,
    brightness_range=[0.1, 0.9],
    fill_mode='nearest'
)

# Creating training set
training_set = train_datagen.flow_from_directory(
    'gameofthrones/train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

# Rescaling the test data between 0 and 1
test_datagen = ImageDataGenerator(rescale=1 / 255)

# Creating test set
test_set = test_datagen.flow_from_directory(
    'gameofthrones/test',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

# Fetching the training set and test set labels
train_labels = list(training_set.class_indices.keys())
test_labels = list(test_set.class_indices.keys())

#No of images in the training set and test set
num_of_train_samples = training_set.samples
num_of_test_samples = test_set.samples

#Steps per epoch and validation steps
steps_per_epoch = num_of_train_samples//batch_size
validation_steps = num_of_test_samples // batch_size


# Building the CNN
# Initialising the CNN
cnn = Sequential()
# Input layer
cnn.add(Input(shape=(img_rows, img_cols, 3)))

# First Convolution Layer
cnn.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))

# First Pooling Layer
cnn.add(MaxPool2D(pool_size=2, strides=2))

# First dropout layer
cnn.add(Dropout(0.25))

# Second convolutional layer
cnn.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))

# Second Pooling Layer
cnn.add(MaxPool2D(pool_size=2, strides=2))

# Second Dropout Layer
cnn.add(Dropout(0.3))


# First Convolution Layer
cnn.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
# Third Pool Layer
cnn.add(MaxPool2D(pool_size=2, strides=2))
# Third Dropout Layer
cnn.add(Dropout(0.3))

# Flattening for the CCN
cnn.add(Flatten())

# Full Connection
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.3))
# Output Layer
cnn.add(Dense(units=4, activation='softmax'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the CNN
print(cnn.summary())

checkpoint = ModelCheckpoint("got15.tf", monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')#save_weights_only=True,
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5, patience=3, verbose=1, mode='max',min_lr=0.00001)
callsback = [checkpoint, reduce_lr]

# Training the CNN on the Training set and evaluating it on the Test set
history = cnn.fit(x=training_set, validation_data=test_set, epochs=15)

#Confution Matrix and Classification Report
Y_pred = cnn.predict(test_set,steps_per_epoch+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')
target_names = list(training_set.class_indices.keys())
print(classification_report(test_set.classes, y_pred, target_names=target_names))
print('Confusion Matrix')
cf_matrix = confusion_matrix(test_set.classes, y_pred)
print(cf_matrix)

# Saving the model
cnn.save('models/gotmodelfinal3.h5')


# # Plotting the accuracy
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

