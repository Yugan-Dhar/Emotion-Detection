# Emotion-Detection

Overview
This project is an Emotion Detection System built using Python and the Keras deep learning library. The system uses a convolutional neural network (CNN) to classify images of faces into one of six emotions: happy, sad, angry, surprised, neutral, and disgusted.

Code Structure
Importing Libraries: The necessary libraries, including Keras, TensorFlow, and OpenCV, are imported.
Loading Data: The dataset of images is loaded and preprocessed.
Building the Model: The CNN model is defined and compiled.
Training the Model: The model is trained on the dataset.
Testing the Model: The model is tested on a separate test dataset.
Results: The results of the model's performance are displayed.
Dataset
The dataset used for this project consists of images of faces, each labeled with one of the six emotions. The dataset is not included in this repository, but can be obtained from [insert link to dataset].

Model Architecture
The model used for this project is a CNN with the following architecture:

Input Layer: Takes in images of size 48x48x1.
Conv2D Layer: 32 filters, kernel size 3x3.
Max Pooling Layer: Pool size 2x2.
Conv2D Layer: 64 filters, kernel size 3x3.
Max Pooling Layer: Pool size 2x2.
Flatten Layer: Flattens the output.
Dense Layer: 128 units, ReLU activation.
Output Layer: 6 units, softmax activation.
Results
The model achieved an accuracy of 0.94 on the test dataset.

Usage
To use this project, simply run the emotion.ipynb notebook in a Jupyter environment. The notebook will load the dataset, build and train the model, and display the results.

Requirements
Python 3.x
Keras 2.x
TensorFlow 2.x
OpenCV 4.x
