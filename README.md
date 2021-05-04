# Age-and-gender-detector-using-machine-learning
This is a machine learning project using python that allows user to detect age and gender

First introducing you with the terminologies used in this  project –

<h5 style="color:green">What is Computer Vision?</h5>
Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

<h5 style="color:green">What is OpenCV?</h5>
OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

<h5 style="color:green">What is a CNN?</h5>
A Convolutional Neural Network is a deep neural network (DNN) widely used for the purposes of image recognition and processing and NLP. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.

<h5 style="color:blue">Objective</h5>
To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using Deep Learning on the Adience dataset.

<h5 style="color:green">About the Project</h5>
In this Python Project, I used Deep Learning to accurately identify the gender and age of a person from a single image of a face.I used the models trained by Tal Hassner and Gil Levi. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.

<h5 style="color:green">The CNN Architecture</h5>
The convolutional neural network for this python project has 3 convolutional layers:

Convolutional layer; 96 nodes, kernel size 7
Convolutional layer; 256 nodes, kernel size 5
Convolutional layer; 384 nodes, kernel size 3
It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, the program will:

Detect faces
Classify into Male/Female
Classify into one of the 8 age ranges
Put the results on the image and display it
The Dataset
For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it here. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.

<h5 style="color:green">Prerequisites</h5>
You’ll need to install OpenCV (cv2) to be able to run this project. You can do this with pip-

<h6 style="color:yellow">pip install opencv-python</h6>
Other packages you’ll be needing are math and argparse, but those come as part of the standard Python library.
<p>For Complete Source Code Kindly Mail Me at <a href="mailto:gordonjohao@gmail.com">gordonjohao@gmail.com</a> At $20 only</p>
