# Object_Detection_And_Classification
The aim of this project/assignment is to build a Deep learning computer vision pipeline for real-time object detection and classification for detecting cars and classifying them as SUV or Sedan after which the processing flow pipeline should be optimized.

## Code Details
## 1 Files:

pipeline.py – The python script for the pipeline \
car_type_classifier.py – The python script used to train the classifier to predict sedan or hatchback using transfer learning \
mobilenet_cars.h5 – This file needs to be generated by running the classifier program \
training_data/ – This directory contains the training data used for the car type classifier \
results.csv – This is the output CSV file containing the model predictions \
video.mp4 – This is the input video from the pipeline \
font/ - This directory contains files required for adding video annotations. \

## 2 Versions:
Installations:
Python Version: 3.7.4 \
pip install tensorflow==1.15 \
pip install keras==2.1.5 \
pip3 install idt 

## 3 Run command:
python pipeline.py <query> Where <query> can be either Q1 , Q2 . Note that the query can be changed during run time by holding down either the 1 or 2 key for a couple of seconds. The changes will be seen in the terminal when a car is in the current frame.

## 4 Notes about pipeline output:
The predictions get stored in results.csv when the pipeline is run for Q2 The videos to accompany the submission are in the output_videos directory

## 5 Further Details:
Cloned YOLO code from: https://github.com/qqwweee/keras-yolo3 Downloaded the Tiny YOLO weights at: https://pjreddie.com/media/files/yolov3-tiny.weights Converted the following to create the Tiny YOLO weights to the correct format: cd keras-yolo3-master python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/tiny_yolo.h5

Made changes to: yolo.py Changed detect_image() method to return information about the bounding box
