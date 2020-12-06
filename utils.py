#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 23:48:00 2020

@author: Ahmed Majuid
"""
import os, shutil

def create_trial_tree(trial_number, session_mode):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # main folder of this trial (holds results, weights and training log)
    trial_root_folder = os.path.join("DeepNav_results", "trial_" + str(trial_number).zfill(3))

    # weights folder
    weights_folder = os.path.join(trial_root_folder, "weights")

    # history csv file
    history_csv_file = os.path.join(trial_root_folder, "model_history_log.csv")

    trial_tree = { "trial_root_folder" : trial_root_folder,
                   "weights_folder" : weights_folder,
                   "history_csv_file" : history_csv_file
                 }

    if session_mode == "Resume":
        return trial_tree
    
    if session_mode == "Override":
        shutil.rmtree(trial_root_folder)   

    if session_mode == "Evaluate":
        shutil.rmtree(os.path.join(trial_root_folder, "training"))
        shutil.rmtree(os.path.join(trial_root_folder, "validation"))
     
    # tree levels
    folders_level_1 = ["training", "validation"]
    folders_level_2 = ["differenced", "reconstructed"]
    folders_level_3 = ["best", "worst", "other"]

    # leaf folders
    for folder_level_1 in folders_level_1:
        for folder_level_2 in folders_level_2:
            for folder_level_3 in folders_level_3:
                leaf_folder = os.path.join(trial_root_folder, folder_level_1, folder_level_2, folder_level_3)
                os.makedirs(leaf_folder)

    # folder for csv files to compare with GPS-less EKF
    for folder_level_1 in folders_level_1:
        output_csv_folder = os.path.join(trial_root_folder, folder_level_1, "reconstructed", "nn_output_csv")
        os.mkdir(output_csv_folder)

    if session_mode == "Evaluate":
        return trial_tree
        
    os.makedirs(weights_folder)
   
    print("\n\n *** \t Created ", trial_root_folder, "\t ***")

    return trial_tree

def retrieve_latest_weights(weights_folder):
    """
    Arguments
        weights_folder: path string of the saved weights directory
    Returns
        last_saved_epoch: integer number, the last saved epoch 
        last_saved_weights_file: string name of the most recent weights file in the directory
    """
    # find the most recent weights and load them to the singeleton model
    saved_weights_names = os.listdir(weights_folder)
    epochs_numbers = []
    for weight_name in saved_weights_names:
        epoch_number = ''.join([i for i in weight_name[2:8] if i.isdigit()])
        epochs_numbers.append(int(epoch_number))

    last_saved_epoch = max(epochs_numbers)
    final_weights_index = epochs_numbers.index(last_saved_epoch)
    final_weights = saved_weights_names[final_weights_index]

    return last_saved_epoch, os.path.join(weights_folder, final_weights)