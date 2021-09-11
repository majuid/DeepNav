#!/usr/bin/env python3
"""
Created on Mon 09 Nov 2020 | 5:25 PM

@author: Ahmed Majuid

Usage: 
Define the network architecture and training hyperparameters
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf messages
import tensorflow as tf
import training
import utils
import postprocessing
from preprocessing.create_dataset import create_dataset

# Session Parameters
session_mode = ["Fresh", "Resume", "Evaluate", "Override"]
mode_id = 0
gpu_name = ["/GPU:0", "/GPU:1", None]
gpu_id = 0
create_new_dataset = True 

# Network Architecture
model_architecture = [
    tf.keras.layers.LSTM(20, return_sequences=True), #200
    # tf.keras.layers.LSTM(200, return_sequences=True),
    # tf.keras.layers.LSTM(200, return_sequences=True),
    tf.keras.layers.LSTM(20, return_sequences=False), #200S
    tf.keras.layers.Dense(6)
    ]

# looping on parameters
varying_hyperparam = None
hyperparam_values = [None]

# Network Hyperparameters
session_data = {"trial_number" : 1,

                "session_mode" : session_mode[mode_id],
                "gpu_name" : gpu_name[gpu_id],

                "batch_size" : int(1 * 1024),
                "learning_rate" : 0.001,
                "window_size" : 10, # 200
                "dropout" : 0.0,
                "epochs" : 3,  # 100
                "initial_epoch" : 0
                }

# create folders for the training outputs (weights, plots, loss history)
trial_tree = utils.create_trial_tree(session_data["trial_number"], session_data["session_mode"])

if create_new_dataset:
    session_data["dataset_name"] = None
    colum_names = {"features"     : ["w_x", "w_y", "w_z", "a_x", "a_y", "a_z", "m_x", "m_y", "m_z"],
                    "features_diff": ["h"],
                    "labels"       : ["Vn", "Ve", "Vd", "Pn", "Pe", "Pd"]}
else:
    session_data["dataset_name"] = "T001_logs548_F10L6_W50_03Dec2020_1542_FMUV5"
    colum_names = {}
    
# create windowed datasets from the flight csv files (or retrieve an old one from binary files)
train_ds, val_dataset, train_flights_dict, val_flights_dict, signals_weights = create_dataset(session_data, colum_names)

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
                                    trial_tree["trial_root_folder"], n_extreme_flights=30)

# add the network configuration and performance to the summary csv
postprocessing.summarize_session(trial_tree, model, session_data, flights_summary)

# save a keras model
keras_model_path = trial_tree["trial_root_folder"] + "/keras_model"
model.save(keras_model_path)

# save the model in tf SavedModel format
tf_model_path = trial_tree["trial_root_folder"] + "/tf_saved_model"
tf.saved_model.save(model, tf_model_path)