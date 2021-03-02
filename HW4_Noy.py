#!/usr/bin/env python
# coding: utf-8

# HW: X-ray images classification


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
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="2"


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
    classes = ['b','c','l','h']
    imagelist = [fn for fn in os.listdir(datapath)]
    N = len(imagelist)
    num_classes = len(classes)
    images = np.zeros((N, 32, 32, 1))
    Y = np.zeros((N,num_classes))
    ii=0
    for fn in imagelist:

        src = imread(os.path.join(datapath, fn),1)
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




# PART 1: Fully connected layers

#--------------------------Impelment your code here:-------------------------------------
# # flatten the images:
#  BaseX_train = BaseX_train.reshape(BaseX_train.shape[0], 32**2)
#  BaseX_val = BaseX_val.reshape(BaseX_val.shape[0], 32**2)
#  X_test = X_test.reshape(X_test.shape[0], 32**2)
# #
#  X_train_orig = X_train_orig.astype('float32')
#  BaseX_train = BaseX_train.astype('float32')
#  BaseX_val = BaseX_val.astype('float32')
# X_test = X_test.astype('float32')
#
# #normalizing the data to help the training:
# # X_train_orig /= 255
# BaseX_train /= 255
# BaseX_val /= 255
# X_test /= 255

# #print the final input shape ready for training:
# # print ("Original Train matrix shape", X_train_orig.shape)
# print ("Train matrix shape", BaseX_train.shape)
# print ("Validation matrix shape", BaseX_val.shape)
# print ("Test matrix shape", X_test.shape)


# Build the model:
input_shape = (32,32,1)
model_relu = Sequential(name="model_relu")
model_relu.add(Flatten(input_shape=input_shape))
model_relu.add(Dense(300, activation='relu', kernel_initializer='he_normal', name='Relu_1')
model_relu.add(Dropout(0.2))

model_relu.add(Dense(150, activation='relu', kernel_initializer='he_normal', name='Relu_2'))
model_relu.add(Dropout(0.2))

#Classification
model_relu.add(Dense(4, activation='softmax', name='Relu_3'))

#----------------------------------------------------------------------------------------

model_relu.summary()


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# Compile and train the model:
#--------------------------Impelment your code here:-------------------------------------
# Compile:
model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='AdamOpt')
## check the'categorical_crossentropy'

## check if needed
# # save initial weights:
# if not("results_1" in os.listdir()):
#     os.mkdir("results_1")
# save_dir = "results_1/"
# model_name = "init_weigths_1"
# model_path = os.path.join(save_dir, model_name)
# model_relu.save(model_path)
# print('Saved initialized model at %s ' % model_path)

# train the model:
history = model_relu.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))

### need to print the final loss??
#----------------------------------------------------------------------------------------



#-------------------------------------Part 2---------------------------------------------------------#
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
new_a_model = Sequential(name="new_a_model")
new_a_model.add(Dense(300, input_shape=(1024,)))
new_a_model.add(Activation('LeakyRelu', name='LeakyRelu_1'))
new_a_model.add(Dropout(0.2))

new_a_model.add(Dense(150))
new_a_model.add(Activation('LeakyRelu', name='LeakyRelu_2'))
new_a_model.add(Dropout(0.2))

### check
new_a_model.add(Dense(10))
new_a_model.add(Activation('softmax'))
#----------------------------------------------------------------------------------------


# In[ ]:


new_a_model.summary()


# ***need to check the diffrences!!

# ---
# <span style="color:red">***Task 3:***</span> *Number of epochs.* 
# 
# Train the new model using 25 and 40 epochs. What difference does it makes in term of performance? Remember to save the compiled model for having initialized weights for every run as we did in tutorial 12. Evaluate each trained model on the test set*
# 
# ---

# In[ ]:


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 25

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# In[ ]:


#--------------------------Impelment your code here:-------------------------------------
# Compile:
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='AdamOpt')
## check the'categorical_crossentropy'

# save initial weights:
if not("results_2" in os.listdir()):
    os.mkdir("results_2")
save_dir2 = "results_2/"
model_name2 = "init_weigths_2"
model_path2 = os.path.join(save_dir2, model_name2)
new_a_model.save(model_path2)
print('Saved initialized model at %s ' % model_path)

# train the model:
history2_25 = model_2.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
#-----------------------------------------------------------------------------------------


# In[ ]:


#Inputs: 
input_shape = (32,32,1)
learn_rate = 1e-5
decay = 0
batch_size = 64
epochs = 40

#Defining the optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# In[ ]:


#--------------------------Impelment your code here:-------------------------------------
# Compile:
new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='AdamOpt')
## check the'categorical_crossentropy'

# save initial weights:
if not("results_2" in os.listdir()):
    os.mkdir("results_2")
save_dir2 = "results_2/"
model_name2 = "init_weigths_2"
model_path2 = os.path.join(save_dir2, model_name2)
new_a_model.save(model_path2)
print('Saved initialized model at %s ' % model_path2)

# train the model:
history2_40 = new_a_model.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
#-----------------------------------------------------------------------------------------


# ---
# <span style="color:red">***Task 4:***</span> *Mini-batches.* 
# 
# Build the `model_relu` again and run it with a batch size of 32 instead of 64. What are the advantages of the mini-batch vs. SGD?*
# 
# ---

# In[ ]:


keras.backend.clear_session()


# In[ ]:


#--------------------------Impelment your code here:-------------------------------------
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
#----------------------------------------------------------------------------------------


# In[ ]:


batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)


# In[ ]:


#--------------------------Impelment your code here:-------------------------------------
# Compile:
model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='AdamOpt')
## check the'categorical_crossentropy'

## check if needed
# # save initial weights:
# if not("results_1" in os.listdir()):
#     os.mkdir("results_3")
# save_dir3 = "results_3/"
# model_name3 = "init_weigths_3"
# model_path3 = os.path.join(save_dir, model_name3)
# model_relu.save(model_path3)
# print('Saved initialized model at %s ' % model_path3)

# train the model:
history3 = model_relu.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
#----------------------------------------------------------------------------------------


# ---
# <span style="color:red">***Task 4:***</span> *Batch normalization.* 
# 
# Build the `new_a_model` again and add batch normalization layers. How does it impact your results?*
# 
# ---

# In[ ]:


keras.backend.clear_session()


# In[ ]:


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
model.add(BatchNormalization())
model_relu.add(Activation('relu', name='Relu_1'))                            
model_relu.add(Dropout(0.2))

model_relu.add(Dense(150))
model.add(BatchNormalization())
model_relu.add(Activation('relu', name='Relu_2'))
model_relu.add(Dropout(0.2))

### check
model_relu.add(Dense(10))
model.add(BatchNormalization())
model_relu.add(Activation('softmax'))

### 4 neurons for classification??


# In[ ]:


batch_size = 32
epochs = 50

#Define your optimizar parameters:
AdamOpt = Adam(lr=learn_rate,decay=decay)
#Compile the network: 


# In[ ]:


#Preforming the training by using fit 
#--------------------------Impelment your code here:-------------------------------------
# Compile:
model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='AdamOpt')
## check the'categorical_crossentropy'

## check if needed
# # save initial weights:
# if not("results_1" in os.listdir()):
#     os.mkdir("results_1")
# save_dir = "results_1/"
# model_name = "init_weigths_1"
# model_path = os.path.join(save_dir, model_name)
# model_relu.save(model_path)
# print('Saved initialized model at %s ' % model_path)

# train the model:
history4 = model_relu.fit(BaseX_train, BaseY_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(BaseX_val, BaseY_val))
#----------------------------------------------------------------------------------------


# ### PART 2: Convolutional Neural Network (CNN)
# ------------------------------------------------------------------------------------

# ---
# <span style="color:red">***Task 1:***</span> *2D CNN.* 
# 
# Have a look at the model below and answer the following:
# * How many layers does it have?
# * How many filter in each layer?
# * Would the number of parmaters be similar to a fully connected NN?
# * Is this specific NN performing regularization?
# 
# ---

# In[ ]:


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
Checkp = ModelCheckpoint(Checkpath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, save_freq=1)


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


# ---
# <span style="color:red">***Task 2:***</span> *Number of filters* 
# 
# Rebuild the function `get_net` to have as an input argument a list of number of filters in each layers, i.e. for the CNN defined above the input should have been `[64, 128, 128, 256, 256]`. Now train the model with the number of filters reduced by half. What were the results.
# 
# ---

# In[ ]:


#--------------------------Impelment your code here:-------------------------------------
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
#----------------------------------------------------------------------------------------


# That's all folks! See you :)
