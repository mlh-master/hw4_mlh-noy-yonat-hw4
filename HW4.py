import os
import numpy as np

import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Activation
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras import utils

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
from skimage.io import imread

from skimage.transform import rescale, resize, downscale_local_mean
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="2"

import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def preprocess(datapath):
    # This part reads the images
    classes = ['b', 'c', 'l', 'h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N, num_classes))
    ii = 0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn), 1)
        img = resize(src, (32, 32), order=3)

        images[ii, :, :, 0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii, cc] = 1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY

def preprocess_train_and_val(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        images[ii,:,:,0] = imread(os.path.join(datapath, fn),1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y

#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)

keras.backend.clear_session()

#--------------------------Impelment your code here:-------------------------------------
# flatten the images:
BaseX_train = BaseX_train.reshape(BaseX_train.shape[0], 32**2)
BaseX_val = BaseX_val.reshape(BaseX_val.shape[0], 32**2)
X_test = X_test.reshape(X_test.shape[0], 32**2)

# X_train_orig = X_train_orig.astype('float32')
BaseX_train = BaseX_train.astype('float32')
BaseX_val = BaseX_val.astype('float32')
X_test = X_test.astype('float32')

#normalizing the data to help the training:
# X_train_orig /= 255
BaseX_train /= 255
BaseX_val /= 255
X_test /= 255

#print the final input shape ready for training:
# print ("Original Train matrix shape", X_train_orig.shape)
print ("Train matrix shape", BaseX_train.shape)
print ("Validation matrix shape", BaseX_val.shape)
print ("Test matrix shape", X_test.shape)


# Build the model:
model_relu = Sequential(name="model_relu")
model_relu.add(Dense(300, input_shape=(1024,)))
model_relu.add(Activation('relu', name='Relu_1'))
model_relu.add(Dropout(0.2))

model_relu.add(Dense(150))
model_relu.add(Activation('relu', name='Relu_2'))
model_relu.add(Dropout(0.2))

### check
model_relu.add(Dense(10))
model_relu.add(Activation('softmax'))

### 4 neurons for classification??
model_relu.summary()

#Inputs:
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

