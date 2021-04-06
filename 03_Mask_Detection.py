# -*- coding: utf-8 -*-
# @Time    : 04/04/2021 17:03
# @Author  : Cao Junhao
# @Site    : 
# @File    : 03_Mask_Detection.py
# @Software: PyCharm
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2

# Construct the argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                type=str,
                default='test_image/02.png',
                help="path to input image")
ap.add_argument("-i1", "--image1",
                type=str,
                default="test_image/03.jpg",
                help="path to input image")
ap.add_argument("-i2", "--image2",
                type=str,
                default='test_image/Junhao_Yuan02.jpg',
                help="path to input image")
ap.add_argument("-i3", "--image3",
                type=str,
                default='test_image/Junhao.jpg',
                help="path to input image")
ap.add_argument("-o", "--output", type=str,
                default="detection_image_04.png",
                help="path to optional output video file")
ap.add_argument("-m", "--model",
                type=str,
                default="DenseNet201_mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-p", "--prototxt",
                default="D:/01_CaoJunhao/FaceDetectSSD/deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-f", "--face_detector",
                type=str,
                default="D:/01_CaoJunhao/FaceDetectSSD/VGG_WIDERFace_SSD_300x300_iter_120000.caffemodel",
                help="path to face detector model directory")
args = vars(ap.parse_args())

# Define the colour corresponding each situation of the mask respectively.
labels_dict = {0: 'Incorrect Mask',
               1: 'Correct Mask',
               2: "No Mask"}
color_dict = {0: (255, 255, 0),
              1: (0, 255, 0),
              2: (0, 0, 255)}


def mask_detection():
    # Loading the face detector training with SSD
    net = cv2.dnn.readNetFromCaffe(args['prototxt'],
                                   args['face_detector'])

    # Loading the mask detector model
    model = load_model(args['model'])

    # Loading the image from Mask_ML, and extract its spatial dimensions
    image = cv2.imread(args['image3'])
    (high, weight) = image.shape[:2]
    print(high)
    print(weight)

    # Construct a blob format for the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), 127.5)

    # Pass the blob through network and acquire the face detections
    print('Extracting the face detections ....')
    net.setInput(blob)
    detections = net.forward()

    # Loop over obtained detections
    for i in range(0, detections.shape[2]):
        # Grab the confidence (probability) associated with the detection
        confidence = detections[0, 0, i, 2]

        # Cutting off the weak detections guarantees the result is great than the minimum probability
        if confidence > args["confidence"]:
            # Confirm the (x,y) to bounding box for each object
            box_object = detections[0, 0, i, 3:7] * np.array([weight,
                                                              high,
                                                              weight,
                                                              high])
            x_start, y_start, x_end, y_end = box_object.astype('int')
            # To ensure the bounding box is inserted into the dimensions of the frame
            (x_start, y_start) = (max(0, x_start-7),
                                  max(0, y_start-7))
            (x_end, y_end) = (min(weight - 1, x_end+7),
                              min(high - 1, y_end+7))

            # Extract the ROI (Region Of Interest)
            # Transform it from BGR to RGB channel
            # Resize it to 224*224, and process it
            face_object = image[y_start: y_end, x_start: x_end]
            # cv2.imshow("Cropped", face_object)
            # cv2.waitKey(0)

            face_object = cv2.cvtColor(face_object, cv2.COLOR_BGR2RGB)
            face_object = cv2.resize(face_object, (224, 224))
            face_object = img_to_array(face_object)
            face_object = preprocess_input(face_object)
            face_object = np.expand_dims(face_object, axis=0)
            print(face_object.shape)
            result = model.predict(face_object)
            label = np.argmax(result, axis=1)[0]
            face_label = labels_dict[label]
            color_label = color_dict[label]

            (Incorrect_Mask, Correct_Mask, No_Mask) = model.predict(face_object)[0]

            # Include the probability in the label
            label = "{}: {:.2f}%".format(face_label,
                                         max(Incorrect_Mask,
                                             Correct_Mask,
                                             No_Mask) * 100)

            cv2.putText(image, label,
                        (x_start, y_start - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, color_label, 2)

            cv2.rectangle(image, (x_start, y_start),
                          (x_end, y_end), color_label, 2)

    cv2.imshow('Image', image)
    key = cv2.waitKey(0)
    cv2.imwrite(args["output"], image)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    mask_detection()

