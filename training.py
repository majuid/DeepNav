#!/usr/bin/env python3
"""
Created on Fri Jun 12 21:48:47 2020

@author: Ahmed Majuid

This script is the training backend, called by training_options.py
"""

import os
import time
import tensorflow as tf

print("tensorflow version = ", tf.__version__)

from utils import retrieve_latest_weights

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.regularizers import l1

# prevent tensorflow from reserving the entire gpu memory
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def start_training(session_data, model_architecture, train_ds, val_ds, signals_weights_tensor, trial_tree):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    epochs = session_data["epochs"]
    initial_epoch = session_data["initial_epoch"]

    weights_folder = trial_tree["weights_folder"]
    history_csv_file = trial_tree["history_csv_file"]

    # create a distribution strategy for multi gpus
    if session_data["gpu_name"] is None:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of GPU devices: {}'.format(strategy.num_replicas_in_sync))

    # Fit Callbacks
    # learning rate scheduling callback
    def scheduler(epoch):
        if epoch < 200:
            return session_data["learning_rate"]
        elif epoch < 300:
            return 0.5 * session_data["learning_rate"]
        else:
            return 0.25 * session_data["learning_rate"]

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # logging callback
    csv_logger = tf.keras.callbacks.CSVLogger(history_csv_file, append=True)

    # weights checkpoint callback
    checkpoint_filepath = weights_folder + '/ep.{epoch:04d}-loss{loss:.4f}-val_loss{val_loss:.4f}.hdf5'
    model_ckpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, period=25)

    callbacks=[csv_logger, model_ckpoint, lr_callback]

    # Retrieve the most recent weights if evaluating or resuming
    if session_data["session_mode"] == "Evaluate":
        epochs = 0
        _, retrieved_weights_file = retrieve_latest_weights(weights_folder)
        print("Evaluating with weights file: ", retrieved_weights_file)
    elif session_data["session_mode"] == "Resume":
        initial_epoch, retrieved_weights_file = retrieve_latest_weights(weights_folder)
        print("Resuming with weights file: ", retrieved_weights_file)

    # Custom Loss Function
    def weighted_MAE(y_true, y_pred):
        """
        Computes mean absolute error between labels and predictions, and weighs it
        such that signals with low values recieve high weights
        """
        y_pred = ops.convert_to_tensor_v2(y_pred)
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(math_ops.multiply(math_ops.abs(y_pred - y_true), signals_weights_tensor), axis=-1)

    start_train_time = time.time()

    if session_data["gpu_name"] is None:
        # distributed model to use multiple GPUs in training
        with strategy.scope():  
            distributed_model = tf.keras.models.Sequential(model_architecture)
            optimizer = tf.keras.optimizers.Adam(lr=session_data["learning_rate"])

            distributed_model.compile(loss=weighted_MAE,
                        optimizer=optimizer,
                        metrics=["mae"])
            
            distributed_model.build(input_shape=(session_data["batch_size"], \
                                                 session_data["window_size"], session_data["n_features"]))
            distributed_model.save_weights(os.path.join(weights_folder, 'ep.0000-loss200-val_loss200.hdf5'))

            if session_data["session_mode"] == "Resume" or session_data["session_mode"] == "Evaluate":
                distributed_model.load_weights(retrieved_weights_file)

            history = distributed_model.fit(train_ds, validation_data=val_ds, epochs=epochs,
            initial_epoch=initial_epoch, callbacks=callbacks)

        # single GPU model for prediction
        singeleton_model = tf.keras.models.Sequential(model_architecture)
        singeleton_model.build(input_shape=(session_data["batch_size"], \
                                            session_data["window_size"], session_data["n_features"]))
        _, retrieved_weights_file = retrieve_latest_weights(weights_folder)
        singeleton_model.load_weights(retrieved_weights_file)

    else:
        with tf.device(session_data["gpu_name"]):  
            singeleton_model = tf.keras.models.Sequential(model_architecture)
            optimizer = tf.keras.optimizers.Adam(lr=session_data["learning_rate"])
            singeleton_model.compile(loss=weighted_MAE, optimizer=optimizer, metrics=["mae"])
            singeleton_model.build(input_shape=(session_data["batch_size"], \
                                                session_data["window_size"], session_data["n_features"]))
            singeleton_model.save_weights(os.path.join(weights_folder, 'ep.0000-loss200-val_loss200.hdf5'))
            if session_data["session_mode"] == "Resume" or session_data["session_mode"] == "Evaluate":
                singeleton_model.load_weights(retrieved_weights_file)

            history = singeleton_model.fit(train_ds, validation_data=val_ds, epochs=epochs,
            initial_epoch=initial_epoch, callbacks=callbacks)

    end_train_time = time.time()
    session_data["training_time_hr"] = (end_train_time - start_train_time) / 3600

    return singeleton_model

if __name__ == '__main__':
    start_training()