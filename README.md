# **Fashion-MNIST Classification with Convolutional Neural Networks**

This is a project that uses Convolutional Neural Networks (CNNs) to classify the Fashion-MNIST dataset. The Fashion-MNIST dataset contains 70,000 grayscale images of 28x28 pixels, each associated with a label from one of 10 classes.

The project is developed in Python, using the TensorFlow and Keras libraries.

## **Prerequisites**

To run this project you need to have the following libraries installed:

- tensorflow
- keras
- IPython
- matplotlib
- scikit-learn

## **Dataset**

The Fashion-MNIST dataset is downloaded directly from the Keras dataset repository, and then split into a training set and a test set. The images in the dataset are normalized so that all the values in the matrices are between 0 and 1.

## **Model Architecture**

The CNN model is defined using the Sequential API from Keras. The model is composed of several layers:

- The first layer is a Conv2D layer with 64 filters of size 3x3 and a stride of 1. The padding is set to "same" to make sure the output has the same dimensions as the input. The input shape is set to (28, 28, 1), the dimensions of the images in the dataset.
- The second layer is a MaxPooling2D layer with a pool size of 2. This reduces the dimensions of the output produced by the previous layer.
- The third layer is a Dropout layer with a rate of 30%. This helps prevent overfitting.
- The fourth layer is another Conv2D layer with 32 filters of size 3x3 and a stride of 1. The padding is set to "same".
- The fifth layer is another MaxPooling2D layer with a pool size of 2.
- The sixth layer is another Dropout layer with a rate of 30%.
- The seventh layer is a Flatten layer, which flattens the output from the previous layer into a one-dimensional array.
- The eighth layer is a Dense layer with 256 neurons and a ReLU activation function.
- The ninth layer is another Dropout layer with a rate of 50%.
- The tenth and final layer is a Dense layer with 10 neurons (one for each class) and a softmax activation function.

The model is compiled using the Adam optimizer and the sparse categorical crossentropy loss function.

## **Results**

The model achieves an accuracy of around 91% on the test set after 10 epochs of training. A classification report and a confusion matrix are also produced, showing the precision, recall, f1-score, and support for each class.

The notebook with the implementation of this project can be found in this repository.
