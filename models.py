# # HW: X-ray images classification
import os
from help_function import preprocess_train_and_val, preprocess, save_model

from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Activation
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt


# ### PART 1: Fully connected layers ## NN with fully connected layers.
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification.
# Use ReLU activation functions for the hidden layers and He_normal for initialization.
# Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*
def create_model_relu(input_shape):
    # --------------------------Implement your code here:-------------------------------------
    model_relu = Sequential(name="model_relu")
    model_relu.add(Flatten(input_shape=input_shape))
    model_relu.add(Dense(300, activation='relu', kernel_initializer='he_normal', name='dense_1'))
    model_relu.add(Dense(150, activation='relu', name='dense_2'))
    model_relu.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    model_relu.summary()
    return model_relu


# Change the activation functions to LeakyRelu or tanh or sigmoid. Name the new model `new_a_model`. Explain how it can affect the model.*
def create_new_a_model(input_shape):
    # --------------------------Implement your code here:-------------------------------------
    new_a_model = Sequential(name="new_a_model")
    new_a_model.add(Flatten(input_shape=input_shape))
    new_a_model.add(Dense(300, kernel_initializer='he_normal', name='dense_1'))
    new_a_model.add(LeakyReLU())
    new_a_model.add(Dense(150, name='dense_2'))
    new_a_model.add(LeakyReLU())
    new_a_model.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    new_a_model.summary()
    return new_a_model


def create_batch_norm_model(input_shape):
    # --------------------------Implement your code here:-------------------------------------
    batch_norm_model = Sequential(name="batch_norm_model")
    batch_norm_model.add(Flatten(input_shape=input_shape))
    batch_norm_model.add(Dense(300, kernel_initializer='he_normal', name='dense_1'))
    batch_norm_model.add(LeakyReLU())
    batch_norm_model.add(BatchNormalization(name='batch_norm_1'))
    batch_norm_model.add(Dense(150, name='dense_2'))
    batch_norm_model.add(LeakyReLU())
    batch_norm_model.add(BatchNormalization(name='batch_norm_2'))
    batch_norm_model.add(Dense(4, activation='softmax', name='dense_3'))
    # ----------------------------------------------------------------------------------------
    batch_norm_model.summary()
    return batch_norm_model


# Compile the model with the optimizer above, accuracy metric and adequate loss for multiclass task.
# Train your model on the training set and evaluate the model on the testing set. # Print the accuracy and loss over the testing set.
def train_and_evaluate_model(model, batch_size, epochs, BaseX_train, BaseY_train, X_test,
                             Y_test, x_val, Y_val):
    # --------------------------Implement your code here:-------------------------------------#
    history = model.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2
                        ,validation_data=(x_val, Y_val))
    loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print("Test Loss is {:.2f} ".format(loss_and_metrics[0]))
    print("Test Accuracy is {:.2f} %".format(100 * loss_and_metrics[1]))

    return history
    # ----------------------------------------------------------------------------------------
