#!/usr/bin/env python
# coding: utf-8

# # HW: X-ray images classification
# --------------------------------------

# Before you begin, open Mobaxterm and connect to triton with the user and password you were give with. Activate the environment `2ndPaper` and then type the command `pip install scikit-image`.

# In this assignment you will be dealing with classification of 32X32 X-ray images of the chest. The image can be classified into one of four options: lungs (l), clavicles (c), and heart (h) and background (b). Even though those labels are dependent, we will treat this task as multiclass and not as multilabel. The dataset for this assignment is located on a shared folder on triton (`/MLdata/MLcourse/X_ray/'`).

# In[1]:


import os
import numpy as np
import tensorflow 
from tensorflow import keras
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, Dropout, Activation
from tensorflow.keras.layers import Flatten, InputLayer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.optimizers import *
from tensorflow.keras import utils

from tensorflow.keras.initializers import Constant
from tensorflow.keras.datasets import fashion_mnist
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import *
import skimage.io
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from tensorflow.keras.optimizers import Adam


# In[ ]:


import tensorflow as tf
config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[ ]:


def preprocess(datapath):
    # This part reads the images
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = skimage.io.imread(os.path.join(datapath, fn), 1)
        img = resize(src,(32,32),order = 3)
        
        images[ii,:,:,0] = img
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    BaseImages = images
    BaseY = Y
    return BaseImages, BaseY


# In[ ]:


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

        images[ii,:,:,0] = skimage.io.imread(os.path.join(datapath, fn), 1)
        cc = -1
        for cl in range(len(classes)):
            if fn[-5] == classes[cl]:
                cc = cl
        Y[ii,cc]=1
        ii += 1

    return images, Y


# In[ ]:


#Loading the data for training and validation:
src_data = '/MLdata/MLcourse/X_ray/'
train_path = src_data + 'train'
val_path = src_data + 'validation'
test_path = src_data + 'test'
BaseX_train , BaseY_train = preprocess_train_and_val(train_path)
BaseX_val , BaseY_val = preprocess_train_and_val(val_path)
X_test, Y_test = preprocess(test_path)


# In[ ]:


keras.backend.clear_session()


# ### PART 1: Fully connected layers 
# --------------------------------------
# Elaborate a NN with 2 hidden fully connected layers with 300, 150 neurons and 4 neurons for classification. Use ReLU activation functions for the hidden layers and He_normal for initialization. Don't forget to flatten your image before feedforward to the first dense layer. Name the model `model_relu`.*

#--------------------------Impelment your code here:-------------------------------------
input_shape = (32,32,1)
model_relu = Sequential(name="model_relu") #Creating the model
model_relu.add(Flatten(input_shape=input_shape)) # Flatting the image
model_relu.add(Dense(300, activation='relu', kernel_initializer='he_normal', name='Relu_1'))
model_relu.add(Dense(150, activation='relu', kernel_initializer='he_normal', name='Relu_2'))
model_relu.add(Dense(4, activation='softmax', name='Relu_3'))
#---------------------------------------------------------------------------------------
model_relu.summary()

#Inputs:
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate, decay=decay)

# Compile & Train the model

# #--------------------------Impelment your code here:-------------------------------------
# Compile:
model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
## check the'categorical_crossentropy'

# # save initial weights:
if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "model_relu_1.h5"
model_path = os.path.join(save_dir, model_name)
model_relu.save(model_path)
print('Saved initialized model at %s ' % model_path)

model_relu_1 = load_model("results/model_relu_1.h5")
# train the model:
history = model_relu_1.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
loss_and_metrics = model_relu_1.evaluate(X_test, Y_test, batch_size=batch_size)
print("Accuracy for Relu 1 is {:.2f} %".format(100 * loss_and_metrics[1]))
print("Loss for Relu 1 is  {:.2f} ".format(loss_and_metrics[0]))
# #----------------------------------------------------------------------------------------
#-------------------Task 2- different activation function--------#

# #--------------------------Impelment your code here:-------------------------------------
new_a_model = Sequential(name="new_a_model") #Creating the model
new_a_model.add(Flatten(input_shape=input_shape)) # Flatting the image
new_a_model.add(Dense(300, kernel_initializer='he_normal', name='sig_1'))
new_a_model.add(LeakyReLU())
new_a_model.add(Dense(150 , kernel_initializer='he_normal', name='sig_2'))
new_a_model.add(LeakyReLU())
new_a_model.add(Dense(4, activation='softmax', name='sig_3'))
# #----------------------------------------------------------------------------------------
new_a_model.summary()

# -------------------------- Task 3 ------------------------------------------------#
#Inputs:
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

# Compile
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
# Save initial weights:
if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "new_a_model_25.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# # train the model:
new_a_model_25 = load_model("results/new_a_model_25.h5")
history_25 = new_a_model_25.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
loss_and_metrics_25 = new_a_model_25.evaluate(X_test, Y_test, batch_size=batch_size)
print("Accuracy of LeakyRelu 25 epochs is {:.2f} %".format(100 * loss_and_metrics_25[1]))
print("Loss of LeakyRelu 25 epochs is {:.2f} ".format(loss_and_metrics_25[0]))

#--------------------new-a-model-40 epochs---------------------#
#Inputs:
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

# #--------------------------Impelment your code here:-------------------------------------
# Compile
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
# Save initial weights:
if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "new_a_model_40.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# # train the model:
new_a_model_40 = load_model("results/new_a_model_40.h5")
history_40 = new_a_model_40.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
loss_and_metrics_40 = new_a_model_40.evaluate(X_test, Y_test, batch_size=batch_size)
print("Accuracy of LeakyRelu 40 epochs is {:.2f} %".format(100 * loss_and_metrics_40[1]))
print("Loss of LeakyRelu 40 epochs is {:.2f} ".format(loss_and_metrics_40[0]))

# #------------------------Task 4: Mini-batches-----------------------#
keras.backend.clear_session()
batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

print("\nrelu second run- batch size: 32, epochs: 50\n")
model_relu_2=load_model("results/model_relu_1.h5")
history_relu_2 = model_relu_2.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
loss_and_metrics_relu_2 = model_relu_2.evaluate(X_test, Y_test, batch_size=batch_size)
print("Accuracy of Relu 2 batch size=32 is {:.2f} %".format(100 * loss_and_metrics_relu_2[1]))
print("Loss of Relu 2 batch size=32 is {:.2f} ".format(loss_and_metrics_relu_2[0]))
# #----------------------------------------------------------------------------------------



#----------------------------------Task 4- Batch Layers------------------------------------ #
keras.backend.clear_session()
new_a_model = Sequential(name="new_a_model") #Creating the model
new_a_model.add(Flatten(input_shape=input_shape)) # Flatting the image
new_a_model.add(Dense(300, kernel_initializer='he_normal', name='sig_1'))
new_a_model.add(LeakyReLU())
new_a_model.add(BatchNormalization())
new_a_model.add(Dense(150 , kernel_initializer='he_normal', name='sig_2'))
new_a_model.add(LeakyReLU())
new_a_model.add(BatchNormalization())
new_a_model.add(Dense(4, activation='softmax', name='sig_3'))
# #----------------------------------------------------------------------------------------
new_a_model.summary()

batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Preforming the training by using fit
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
# Save initial weights:
if not("results" in os.listdir()):
    os.mkdir("results")
save_dir = "results/"
model_name = "new_a_model_norm.h5"
model_path = os.path.join(save_dir, model_name)
new_a_model.save(model_path)
print('Saved initialized model at %s ' % model_path)

# # train the model:
new_a_model_norm = load_model("results/new_a_model_norm.h5")
history_norm = new_a_model_norm.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
loss_and_metrics_norm = new_a_model_norm.evaluate(X_test, Y_test, batch_size=batch_size)
print("Accuracy with batch normalization is {:.2f} %".format(100 * loss_and_metrics_norm[1]))
print("Loss with batch normalization is {:.2f} ".format(loss_and_metrics_norm[0]))



#-------------------------------------------------------------------------------------------------------------------#






# # ### PART 2: Convolutional Neural Network (CNN)

def get_net(input_shape,drop,dropRate,reg):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:
    model.add(Dense(512, activation='elu',name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model


# In[ ]:


input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
NNet = get_net(input_shape,drop,dropRate,reg)


# In[ ]:


NNet=get_net(input_shape,drop,dropRate,reg)


# In[ ]:


from tensorflow.keras.optimizers import *
import os
from tensorflow.keras.callbacks import *

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network:
NNet.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
Checkpath = os.getcwd()
# Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)


# In[ ]:


#Preforming the training by using fit
# IMPORTANT NOTE: This will take a few minutes!
h = NNet.fit(x=BaseX_train, y=BaseY_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0, validation_data = (BaseX_val, BaseY_val), shuffle=True)
#NNet.save(model_fn)


# In[ ]:


# NNet.load_weights('Weights_1.h5')


# In[ ]:


results = NNet.evaluate(X_test,Y_test)
print('test loss, test acc:', results)



#--------------------------Impelment your code here:-------------------------------------
def get_net_half(input_shape,drop,dropRate,reg,filters):
    #Defining the network architecture:
    model = Sequential()
    model.add(Permute((1,2,3),input_shape = input_shape))
    model.add(Conv2D(filters=filters[0], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_1',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters[1], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_2',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters[2], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_3',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=filters[3], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_4',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(Conv2D(filters=filters[4], kernel_size=(3,3), padding='same', activation='relu',name='Conv2D_5',kernel_regularizer=regularizers.l2(reg)))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #Fully connected network tail:
    model.add(Dense(512, activation='elu',name='FCN_1'))
    if drop:
        model.add(Dropout(rate=dropRate))
    model.add(Dense(128, activation='elu',name='FCN_2'))
    model.add(Dense(4, activation= 'softmax',name='FCN_3'))
    model.summary()
    return model

input_shape = (32,32,1)
learn_rate = 1e-5
decay = 1e-03
batch_size = 64
epochs = 25
drop = True
dropRate = 0.3
reg = 1e-2
filters= [32, 64, 64, 128, 128]
print(filters)
NNet_half = get_net_half(input_shape,drop,dropRate,reg,filters)

# #Defining the optimizar parameters:
# AdamOpt = Adam(lr=learn_rate,decay=decay)

#Compile the network:
NNet_half.compile(optimizer=AdamOpt, metrics=['acc'], loss='categorical_crossentropy')

#Saving checkpoints during training:
Checkpath = os.getcwd()
# Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)
results_half = NNet_half.evaluate(X_test,Y_test)
print('test loss half filters, test acc half filters:', results_half)
#----------------------------------------------------------------------------------------


# That's all folks! See you :)
