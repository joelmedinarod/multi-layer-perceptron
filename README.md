# Multi-Layer Perceptron

Multi-Layer Perceptron (MLP) developed using Numpy.
This package implements the model of a basic Feed-Fordward Neural Network with two hidden layers and configurable number of neurons pro layer. Check **multi_layer_perceptron/model.py**
My goal was to better understand the mathematics behind how neural network works and evaluate how MLPs perform in simple classification tasks, including image-recognition (even though CNNs are more capable for this task). In this package, the model is used for:
 1. Learning a non-linear boundary to separate two classes of datapoints in a 2D Space. **multi_layer_perceptron/circles.py**
 2. Classifying flowers (IRIS Dataset) **multi_layer_perceptron/iris.py**
 3. Classifying images of numbers from 0 to 9 (MNIST Dataset) **multi_layer_perceptron/mnist.py**
 4. Classifying images of clothes (Fashion MNIST Dataset) **multi_layer_perceptron/fashion_mnist.py**

Models for classifying images are saved in models/ directory after being trained. These trained models are used for inference for the following tasks:
 1. Classifying images of numbers from 0 to 9 (MNIST Dataset) **multi_layer_perceptron/load_mnist.py**
 2. Classifying images of clothes (Fashion MNIST Dataset) **multi_layer_perceptron/load_fashion.py**

The implementation of MultiLayerPerceptron is in **multi_layer_perceptron/model.py**. Helper functions are in **multi_layer_perceptron/helper_functions.py**

## Dependencies

python = "3.10.11"
numpy = "1.26.4"
matplotlib = "3.9.0"
scikit-learn = "1.5.0"
tensorflow = "2.16.1"
PyQt6 = "6.7.0"

## Installation

There is a problem installing Tensorflow using Poetry. For this reason, I install all dependencies globally and run the scripts individually. I use tensorflow.keras.datasets to get mnist dataset for classification.

# Usage

I run the scripts from main directory:

To train and inference of circles dataset
```
python multi_layer_perceptron/circles.py
```
To train and inference of IRIS dataset
```
python multi_layer_perceptron/iris.py
```
To train model on MNIST Dataset and save the trained model
```
python multi_layer_perceptron/mnist.py
```
To train model on Fashion MNIST Dataset and save the trained model
```
python multi_layer_perceptron/fashion_mnist.py
```
To load model trained on MNIST Dataset and use it for inference on test data
```
python multi_layer_perceptron/load_mnist.py
```
To load model trained on Fashion MNIST Dataset and use it for inference on test data
```
python multi_layer_perceptron/load_fashion.py
```