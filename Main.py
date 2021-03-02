import os
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from models import create_new_a_model, create_model_relu, train_and_evaluate_model, create_batch_norm_model
from help_function import preprocess_train_and_val, preprocess, save_model, plot_train_stat

if __name__ == '__main__':
    mpl.rc('axes', labelsize=14)
    mpl.rc('xtick', labelsize=12)
    mpl.rc('ytick', labelsize=12)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)
    keras.backend.clear_session()

    # Loading the data for training and validation:
    src_data = '/MLdata/MLcourse/X_ray/'
    train_path = src_data + 'train'
    val_path = src_data + 'validation'
    test_path = src_data + 'test'
    BaseX_train, BaseY_train = preprocess_train_and_val(train_path)
    BaseX_val, BaseY_val = preprocess_train_and_val(val_path)
    X_test, Y_test = preprocess(test_path)

    # Inputs:
    input_shape = (32, 32, 1)
    learn_rate = 1e-5
    decay = 0
    batch_size = 64
    epochs = 25

    # Define your optimizer parameters:
    AdamOpt = Adam(lr=learn_rate, decay=decay)

    # # create model relu - Q1
    # model_relu = create_model_relu(input_shape)
    # model_relu.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    # save_model(model_relu, "model_relu.h5")
    # print("\nload model relu for first run")
    # model_relu_run_1 = load_model("results/model_relu.h5")
    # train_result_relu_1 = train_and_evaluate_model(model_relu_run_1, batch_size, epochs, BaseX_train, BaseY_train, X_test,
    #                                         Y_test, BaseX_val, BaseY_val)
    # plot_train_stat(train_result_relu_1)

    # create new_a_model - Q2
    new_a_model = create_new_a_model(input_shape)
    new_a_model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    save_model(new_a_model, "new_a_model.h5")

    # Number of epochs - Q3
    epochs1 = 25
    epochs2 = 40
    print("\nload new_a_model for first run with 25 epochs")
    new_model_run_1 = load_model("results/new_a_model.h5")
    new_model_train_result_1 = train_and_evaluate_model(new_model_run_1, batch_size, epochs1, BaseX_train, BaseY_train,
                                                        X_test, Y_test, BaseX_val, BaseY_val)
    plot_train_stat(new_model_train_result_1)
    print("\nload new_a_model for second run with 40 epochs")
    new_model_run_2 = load_model("results/new_a_model.h5")
    new_model_train_result_2 = train_and_evaluate_model(new_model_run_2, batch_size, epochs2, BaseX_train, BaseY_train,
                                                        X_test, Y_test, BaseX_val, BaseY_val)
    plot_train_stat(new_model_train_result_2)
    # Mini-Batches - Q4
    batch_size2 = 32
    epochs3 = 50
    keras.backend.clear_session()
    print("\nload model relu for second run with 32 batch_size and 50 epochs\n")
    model_relu_run_2 = load_model("results/model_relu.h5")
    train_and_evaluate_model(model_relu_run_2, batch_size2, epochs3, BaseX_train, BaseY_train, X_test, Y_test,
                             BaseX_val, BaseY_val)

    # Batch normalization - Q5
    model_batch_norm = create_batch_norm_model(input_shape)
    model_batch_norm.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=AdamOpt)
    # save_model(model_batch_norm, "model_batch_norm.h5")
    # print("\nload model batch norm for run")
    # model_batch_norm_run = load_model("results/model_batch_norm.h5")
    train_and_evaluate_model(model_batch_norm, batch_size2, epochs3, BaseX_train, BaseY_train, X_test, Y_test,
                             BaseX_val, BaseY_val)
