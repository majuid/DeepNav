#!/usr/bin/env python3
"""
Created on Mon 09 Nov 2020 | 5:25 PM

@author: Ahmed Majuid

Usage: Define the network architecture and training hyperparameters
"""

# non standard library
import os
import tensorflow as tf
from tensorflow.python.keras.backend import variable
import training
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset

# Session Parameters
session_mode = ["Fresh", "Resume", "Evaluate", "Override"]
mode_id = 0
gpu_name = ["/GPU:0", "/GPU:1", None]
gpu_id = 0
create_new_dataset = False 

# Network Architecture
model_architecture = [
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dense(6)
    ]

# looping on parameters
varying_hyperparam = "learning_rate"
hyperparam_values = [0.001, 0.005, 0.01, 0.05, 0.1]

for trial_offset, hyperparam_value in enumerate(hyperparam_values):

    tf.keras.backend.clear_session()

    # Network Hyperparameters
    session_data = {"trial_number" : 8,

                    "session_mode" : session_mode[mode_id],
                    "gpu_name" : gpu_name[gpu_id],

                    "batch_size" : int(2 * 1024),
                    "learning_rate" : 0.005,
                    "window_size" : 50,
                    "dropout" : 0.0,
                    "epochs" : 100,
                    "initial_epoch" : 0,

                    "n_features" : 10,
                    "n_labels" : 6,
                    }

    session_data[varying_hyperparam] = hyperparam_value
    
    session_data["trial_number"] += trial_offset

    print(session_data)

    # create folders for the training outputs (weights, plots, loss history)
    trial_tree = utils.create_trial_tree(session_data["trial_number"], session_data["session_mode"])

    if create_new_dataset:
        session_data["dataset_name"] = None
    else:
        session_data["dataset_name"] = "T001_logs548_F10L6_W50_16Nov2020_0531"
        
    # create windowed datasets from the flight csv files (or retrieve an old one from binary files)
    train_ds, val_dataset, train_flights_dict, val_flights_dict, signals_weights = create_dataset(session_data)

    # batch and shuffle
    train_dataset = train_ds.batch(session_data["batch_size"]).shuffle(buffer_size=1000)
    val_dataset = val_dataset.batch(session_data["batch_size"]).shuffle(buffer_size=1000)

    # print the shape of a single batch
    for x, y in train_dataset.take(1):
        print("\nshape of a single training batch")
        print(x.shape, y.shape)

    # convert signals weights to a tensor to be used by the loss function
    signals_weights_tensor = tf.constant(signals_weights, dtype=tf.float32)

    # start training
    model = training.start_training(session_data, model_architecture, train_dataset, val_dataset, \
                                    signals_weights_tensor, trial_tree)

    # for every flight, plot all states (truth vs predictions)
    flights_summary = postprocessing.evaluate_all_flights(model, train_flights_dict, val_flights_dict, \
                                        trial_tree["trial_root_folder"], n_extreme_flights=10)

    # add the network configuration and performance to the summary csv
    postprocessing.summarize_session(trial_tree, model, session_data, flights_summary)