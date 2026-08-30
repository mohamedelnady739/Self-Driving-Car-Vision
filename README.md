Self Driving Car Vision System 

A Computer Vision project for autonomous driving scenarios using YOLOv8s.

Features

Object Detection

Image Detection

Video Detection

Object Tracking with ByteTrack/BoT-SORT

Vehicle Counting

Pedestrian and Biker Counting

Traffic Light Detection

Streamlit Dashboard

Dataset

Self Driving Car Dataset from Roboflow.

29,800 images

11 classes

YOLO format

Classes:
biker, car, pedestrian, trafficLight, trafficLight-Green,
trafficLight-GreenLeft, trafficLight-Red, trafficLight-RedLeft,
trafficLight-Yellow, trafficLight-YellowLeft, truck

Model

YOLOv8s

Training:

Epochs: 10

Image Size: 640

Batch Size: 16

Results

Test Set:

Precision: 84.4%

Recall: 61.7%

mAP@50: 72.8%

mAP@50-95: 41.1%

Streamlit App

The application supports image detection, video detection, tracking, counting, and a detection dashboard.


📁 Full Project:
[(https://drive.google.com/drive/folders/1wg3ufEsJMX5L4OXMa2YyUbAYIYmGWhnB?usp=sharing)]

Dataset Source

Roboflow Self Driving Car Dataset:
https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/dataset/3
