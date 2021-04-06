# -*- coding: utf-8 -*-
# @Time    : 04/04/2021 21:42
# @Author  : Cao Junhao
# @Site    : 
# @File    : 02_Test_DenseNet-Model.py
# @Software: PyCharm

import tensorflow as tf
import argparse
from imutils import paths
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

# Constructs the environment to fit the Tensorflow_GPU
physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)

# Constructs the argument parse to  parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model',
                    type=str,
                    default="DenseNet201_mask_detector.model",
                    help="path to output face mask detector model")
parser.add_argument("-dataset03", "--test",
                    type=str,
                    default="D:/01_CaoJunhao/Final_Version_Dataset/Test_Dataset")
args = vars(parser.parse_args())
# prints the arguments that we configured
print(args)

# Process the dataset
paths_images = paths.list_images(args['test'])
data = []
true_labels = []
pred_labels = []

# Load the model from the disk
model = load_model(args['model'])
model.summary()

# Label for the class
label_dic = {0: '01_IMFD',
             1: '02_CMFD',
             2: '03_NoMask'}

# Loop over the whole dataset
for path_image in paths_images:
    # Extract the label from each class
    label = path_image.split(os.path.sep)[1]
    true_labels.append(label)
    print(label)

    # Load the image and resize it into 224 x 224
    img = load_img(path=path_image, target_size=(224, 224))
    # Convert it into numpy array
    img_array = img_to_array(img=img)
    # Process input
    image_array = preprocess_input(img_array)
    # Append a extra dimensionality
    image_array = np.expand_dims(a=image_array, axis=0)
    # Predict the probability associated with each class
    pred_probability = model.predict(image_array)
    print(pred_probability)
    # Obtain the predicted label
    label_index = np.argmax(a=pred_probability, axis=1)[0]
    pred_label = label_dic[label_index]
    pred_labels.append(pred_label)


target_names = ['01_IMFD', '02_CFMD', '03_NoMask']

print(classification_report(y_true=true_labels,
                            y_pred=pred_labels,
                            digits=5,
                            target_names=target_names))


