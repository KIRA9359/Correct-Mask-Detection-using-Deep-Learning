# -*- coding: utf-8 -*-
# @Time    : 05/04/2021 21:29
# @Author  : Cao Junhao
# @Site    : 
# @File    : 04_Mask_Detection_Real-Time-Stream.py
# @Software: PyCharm

import tensorflow as tf
import cv2
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.models import load_model
import imutils
from imutils.video import VideoStream

# Construct the environment to fit the Tensorflow_GPU
physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
    tf.config.experimental.set_memory_growth(p, True)

# Construct the argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt",
                default="D:/01_CaoJunhao/FaceDetectSSD/deploy.prototxt",
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-f", "--face_detector",
                type=str,
                default="D:/01_CaoJunhao/FaceDetectSSD/VGG_WIDERFace_SSD_300x300_iter_120000.caffemodel",
                help="path to face detector model directory")
ap.add_argument("-m", "--model",
                type=str,
                default='DenseNet201_new_mask_detector.model',
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print(args)

# Load the face detector from the disk
face_net = cv2.dnn.readNetFromCaffe(prototxt=args['prototxt'],
                                    caffeModel=args['face_detector'])
print(face_net)

# Load the mask detection model from the disk
mask_model = load_model(args['model'])

# Define the colour corresponding each situation of the mask respectively.
labels_dict = {0: 'Incorrect Mask',
               1: 'Correct Mask',
               2: "No Mask"}
color_dict = {0: (255, 255, 0),
              1: (0, 255, 0),
              2: (0, 0, 255)}


# Define a methode to predict the face_mask object
def detect_predict_mask(frame, face_net, mask_net):
    # Acquire the dimensions of the frame and construct the blob object
    print('frame shape : {}'.format(frame.shape))
    high, weight = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(image=frame,
                                 scalefactor=1.0,
                                 size=(300, 300),
                                 mean=(104.0, 177.0, 123.0))
    # Pass the blob through the network and obtain the face detections
    face_net.setInput(blob)
    detections = face_net.forward()
    print('detections shape : ', detections.shape)
    print('detections shape[2] : ', detections.shape[2])
    print(detections)

    # Initialize the list of faces object, their corresponding locations, and the one of predictions from the face mask
    faces = []
    locations = []
    predictions = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        # Extract the confidence associated with the detection
        confidence = detections[0, 0, i, 2]

        # Cut off weak detections by guaranteeing confidence is greater than the minimum confidence
        if confidence > args['confidence']:
            # Compute the (x,y) -- coordinates of the bounding box for each object
            bounding_box_object = detections[0, 0, i, 3:7] * np.array([weight, high, weight, high])
            start_x, start_y, end_x, end_y = bounding_box_object.astype('int')

            # Guaranteeing the bounding boxes fall within the frame
            (start_x, start_y) = (max(0, start_x - 7), max(0, start_y - 7))
            (end_x, end_y) = (min(weight, end_x + 7), min(high, end_y + 7))

            # Extract the ROI (Region of interest)
            face = frame[start_y:end_y, start_x:end_x]
            # Convert it from BGR to RGB
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # Resize it to 224*224
            face = cv2.resize(face, (224, 224))
            # Transform face into numpy array
            face = img_to_array(face)
            # processing it
            face = preprocess_input(face)
            print(face)
            # Add the face and the location of the bounding box to respective list
            faces.append(face)
            locations.append((start_x, start_y, end_x, end_y))

    # To make the prediction is only in the case of at least existing one face.
    if len(faces) > 0:
        faces = np.array(faces, dtype='float32')
        predictions = mask_net.predict(faces, batch_size=32)

    return locations, predictions


#  Loop over the frames from video stream
vs = VideoStream(src=0).start()
# Initialize the video stream and allow it to warm up
while True:
    # Acquire the frame from the threaded video stream
    video_stream_read = vs.read()
    # Resize it to change into a maximum width of 400 pixels
    frame = imutils.resize(video_stream_read, width=600)

    # Detect the face object in the frame and determine whether or not the face object is wearing the mask
    locations, predictions = detect_predict_mask(frame=frame, face_net=face_net, mask_net=mask_model)

    # Traverse the detected face object and corresponding locations
    for (bounding_box, prediction) in zip(locations, predictions):
        # Unbox the bounding box and prediction
        start_x, start_y, end_x, end_y = bounding_box
        print('prediction : {} '.format(prediction))
        print(type(prediction))
        (Incorrect_Mask, Correct_Mask, No_Mask) = prediction
        label_index = np.argmax(prediction)
        face_label = labels_dict[label_index]
        color_label = color_dict[label_index]

        # Append the probability into the label
        label = '{}: {:.4f}%'.format(face_label, max(Incorrect_Mask, Correct_Mask, No_Mask) * 100)

        # Display the label and bounding box on the output frame
        cv2.putText(img=frame,
                    text=label,
                    org=(start_x, start_y - 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50,
                    color=color_label,
                    thickness=2)
        cv2.rectangle(img=frame,
                      pt1=(start_x, start_y),
                      pt2=(end_x, end_y),
                      color=color_label,
                      thickness=2)
    # Show the frame
    cv2.imshow('MyFrame', frame)
    ff = 0xFF
    key = cv2.waitKey(1) & ff

    if key == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

