# -*- coding: utf-8 -*-
# @Time    : 02/04/2021 23:23
# @Author  : Cao Junhao
# @Site    : 
# @File    : 01_DenseNet_Train_Mask_Detector.py
# @Software: PyCharm

import tensorflow as tf
import argparse
from imutils import paths
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constructs the environment to fit the Tensorflow_GPU
physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)

# Constructs the argument parse to  parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',
                    type=str,
                    default="DenseNet201_new_mask_detector.model",
                    help="path to output face mask detector model")
parser.add_argument("-dataset01", "--train",
                    type=str,
                    default="D:/01_CaoJunhao/Final_Version_Dataset/Train_Dataset")
parser.add_argument("-dataset02", "--validation",
                    type=str,
                    default="D:/01_CaoJunhao/Final_Version_Dataset/Validation_Dataset")
args = vars(parser.parse_args())
# prints the arguments that we configured
print(args)

# Initializing the learning rate, Batch Size, and Epochs.
learning_rate = 1e-4
batch_size = 32
epochs = 25


# Loads the dataset and processes it
def load_dataset_et_process(dataset_path):
    paths_images = paths.list_images(dataset_path)
    data = []
    labels = []

    # Loops over the whole dataset
    for image_path in paths_images:
        # Extracts the label from each class
        label = image_path.split(os.path.sep)[1]
        labels.append(label)

        # To load the image resizes it into 224x224
        img = load_img(image_path, target_size=(224, 224))
        # Converts a PIL to a Numpy array
        img_array = img_to_array(img)
        # Preprocesses the input image
        image_array = preprocess_input(img_array)
        data.append(image_array)

    data = np.array(data, dtype='float32')
    labels = np.array(labels)

    print(data.shape)
    print(labels.shape)
    return data, labels


# Performs one-hot encoding on each label
lb = LabelBinarizer()
x_train, y_train = load_dataset_et_process(args['train'])
x_valid, y_valid = load_dataset_et_process(args['validation'])

# Converts it into binary format
y_train = lb.fit_transform(y_train)
y_valid = lb.fit_transform(y_valid)

print('x_train shape : ', x_train.shape)
print('y_train shape : ', y_train.shape)
print('x_validation shape : ', x_valid.shape)
print('y_validation shape : ', y_valid.shape)

# Starting to construct the model
print('Starting to construct model')
# Loading the DenseNet model pre_trained on the imageNet, simultaneously cut off the original FC_Layer
base_model = DenseNet201(weights='imagenet',
                         include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

# It is compulsory for the base_model to freeze each layer,
# which result will never train again in the network.
for layer in base_model.layers:
    layer.trainable = False

# To construct the our own ful_connection layer will be placed on top of the base model
head_model = base_model.output
head_model = MaxPooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(64, activation='relu')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(3, activation='softmax')(head_model)

# Place the head_model on the top of the base model
model = Model(inputs=base_model.input,
              outputs=head_model)

# Compile model
print("compiling model...")
adam_optimize = Adam(learning_rate=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam_optimize,
              metrics=['accuracy'])

# Train the configured FC_layer
model.fit(x=x_train,
          y=y_train,
          batch_size=batch_size,
          validation_data=(x_valid, y_valid),
          epochs=epochs)

# Serialize the model to the disk
print('saving mask detector model...')
model.save(args['model'], save_format='h5')
