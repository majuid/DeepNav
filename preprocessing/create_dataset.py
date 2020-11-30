#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 7 15:48:47 2020

@author: Ahmed Majuid
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
import pickle
import glob

def create_dataset(session_data, colum_names):
    """
    Arguments
        session_data: A dictionary containing the following relevant values
            trial: integer trial number, used to name the dataset folder
            
            n_features: integer number of features, (usually 3 accel, 3 gyro, 3 mag, 1 baro :=> 10)
            
            n_labels: integer number of labels, (usually 4 attitude, 3 velocity, 3 position :=> 10)
            
            window_size: how many timesteps are consumed before one prediction is made
            
            dataset_name: string name of the folder containing the dataset to be retrieved, 
                        a new dataset is created if None

    Return

        flights_dictionaries["training"], flights_dictionaries["validation"], signals_weights

        training_dataset: tf dataset object created from (features, labels) arrays which are
                          built by vertically stacking then shuffling all the windowed features
                          of all flights (n, window_size, n_features)
        
        validation_dataset: similar but for validation

        training_flights_dictionary: a dictionary whose keys are flight names (numbers) and
                                     values are windowed features and labels of each flight
                                     this is used by the evaluation code

        validation_flights_dictionary: same but for validation

        signals_weights: np array of shape (n_labels,1), used to modify the loss function
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    data_directory = os.path.join(os.path.pardir, "DeepNav_data")
    
    # training and validation folders
    sets_subdirs = ["training", "validation"]

    # if the dataset already windowed and saved, load it
    if session_data["dataset_name"] is not None:
        print("retrieving", session_data["dataset_name"])

        datasets_directory = os.path.join(data_directory, "datasets", session_data["dataset_name"])

        # load the training & validation arrays from file (features, labels)
        with open(os.path.join(datasets_directory, "features_labels"), 'rb') as features_labels_file:
            npzfile = np.load(features_labels_file)
            combined_windowed_features = {"training":npzfile["features_tr"], "validation":npzfile["features_val"]}
            combined_windowed_labels = {"training":npzfile["labels_tr"], "validation":npzfile["labels_val"]}

        session_data["n_features"] = combined_windowed_features["validation"].shape[-1]
        # load the flights dictionaries
        with open(os.path.join(datasets_directory, "flights_dictionaries"), 'rb') as flights_dict_file:
            flights_dictionaries = pickle.load(flights_dict_file)

    # if no windowed dataset available, window the csv files 
    else:

        csvs_root_directory = os.path.join(data_directory, "combined_csvs", "trimmed")
        
        # count the csv files in both training and validation subdirectories
        n_logs = len([filename for filename in glob.iglob(csvs_root_directory + '/*/*', recursive=True)])
        session_data["n_features"] = len(colum_names["features"]) + len(colum_names["features_diff"])
        n_features = str(session_data["n_features"])
        n_labels = str(len(colum_names["labels"]))
        
        # a dataset name contains trial number, number of features, labels & logs, window size & time of creation
        dataset_name = "T" + str(session_data["trial_number"]).zfill(3) + "_logs" + str(n_logs) + \
                       "_F" + n_features + "L" + n_labels + "_W" + str(session_data["window_size"]) + \
                       "_" + datetime.datetime.now().strftime("%d%b%Y") + \
                       "_" + datetime.datetime.now().strftime("%H%M")

        datasets_directory = os.path.join(data_directory, "datasets", dataset_name)

        if not os.path.isdir(datasets_directory):
            os.makedirs(datasets_directory)
        
        print("creating", dataset_name, "...")
        session_data["dataset_name"] = dataset_name

        # dictionary of two dictionaries, (training & validation), each subdictionary is a list of windows
        combined_windowed_features = {}
        combined_windowed_labels = {}
        
        # dictionary of two dictionaries, (training & validation), each subdictionary is key-value pairs
        # of (flight_name, list of windows)
        flights_dictionaries = {"training":{}, "validation":{}}

        for set_subdir in sets_subdirs:
            csvs_directory = os.path.join(csvs_root_directory, set_subdir)

            # each element of the list is all the windows of one flight
            x_list = []
            y_list = []

            for flight_file in sorted(os.listdir(csvs_directory)):
                
                # read flight data from csv to features and labels dataframes
                csv_file_name = os.path.join(csvs_directory, flight_file)
                features = pd.read_csv(csv_file_name, usecols=colum_names["features"]).to_numpy()[1:,:]
                features_diff = pd.read_csv(csv_file_name, usecols=colum_names["features_diff"]).to_numpy()
                labels = pd.read_csv(csv_file_name, usecols=colum_names["labels"]).to_numpy()
                
                # apply differencing to labels
                features_diff = np.diff(features_diff, axis=0)
                labels = np.diff(labels, axis=0)

                # stack differenced and non-differecnced features
                features = np.hstack((features,features_diff))
                
                windowed_features = []
                windowed_labels = []
                
                # move a window w on the data, features are at i -> i+w & label is at i+w
                for i in range (labels.shape[0] - session_data["window_size"]):
                    
                    one_window = features[i:i+session_data["window_size"], :]
                    one_label = labels[i+session_data["window_size"], :]
                    
                    windowed_features.append(one_window)
                    windowed_labels.append(one_label)
            
                x_one_flight = np.array(windowed_features)
                y_one_flight = np.array(windowed_labels)

                flights_dictionaries[set_subdir].update({flight_file[0:-4]:(x_one_flight, y_one_flight)})
                
                x_list.append(x_one_flight)
                y_list.append(y_one_flight)

            # vertically stack the feature and labels vectors of all flights
            combined_windowed_features[set_subdir] = np.vstack(x_list)
            combined_windowed_labels[set_subdir] = np.vstack(y_list)

            # shuffle features and labels together
            shuffled_indices = np.arange(combined_windowed_features[set_subdir].shape[0])
            np.random.shuffle(shuffled_indices)
            combined_windowed_features[set_subdir] = combined_windowed_features[set_subdir][shuffled_indices]
            combined_windowed_labels[set_subdir] = combined_windowed_labels[set_subdir][shuffled_indices] 

        # save the dataset to files (features, labels)
        with open(os.path.join(datasets_directory, "features_labels"), 'wb') as features_labels_file:
            np.savez(features_labels_file, \
                     features_tr=combined_windowed_features["training"], \
                     labels_tr=combined_windowed_labels["training"], \
                     features_val=combined_windowed_features["validation"], \
                     labels_val=combined_windowed_labels["validation"])

        # save the flights dictionaries
        with open(os.path.join(datasets_directory, "flights_dictionaries"), 'wb') as flights_dict_file:
            pickle.dump(flights_dictionaries, flights_dict_file,  protocol=pickle.HIGHEST_PROTOCOL)

        # save the colum names (what are the features, labels and what features are diffferenced)
        with open(os.path.join(datasets_directory, "features_labels_names.txt"), 'w') as f:
            for key, value in colum_names.items():
                line = key + ":\n" + ','.join(map(str, value)) + "\n"
                f.write(line)
        
    # create weights to pay more attention for smaller signals when training
    average_absolutes = np.mean(np.abs(combined_windowed_labels["training"]), axis = 0)
    signals_weights = 1 / average_absolutes
    signals_weights = signals_weights / np.min(signals_weights)
    print("\nsignals weights:\n", signals_weights, "\n")

    # print windowing outputs shapes and durations
    for set_subdir in sets_subdirs:
        print("shape of", set_subdir, "features", combined_windowed_features[set_subdir].shape)
        print("shape of", set_subdir, "labels", combined_windowed_labels[set_subdir].shape)
        print("----")

        # Calculate and print the total flight time of this set
        dt = 0.2
        flight_time_hr = f'{(combined_windowed_features[set_subdir].shape[0] * dt / 3600):.2f}'
        print("flight time used for", set_subdir, " : ", flight_time_hr, "hours")
        print("------------")

        session_data_new_key = "flight_duration_" + set_subdir + "_hr"

        session_data[session_data_new_key] = flight_time_hr

    # create a tf dataset object
    training_dataset = tf.data.Dataset.from_tensor_slices((combined_windowed_features["training"], combined_windowed_labels["training"]))
    validation_dataset = tf.data.Dataset.from_tensor_slices((combined_windowed_features["validation"], combined_windowed_labels["validation"]))

    return training_dataset, validation_dataset, flights_dictionaries["training"], flights_dictionaries["validation"], signals_weights

if __name__ == '__main__':
    create_dataset()