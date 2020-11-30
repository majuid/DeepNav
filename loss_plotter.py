#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 21:48:47 2020

@author: Ahmed Majuid
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# trials you want to plot as a comma separated string (ex. 73,74,75)
trials_str = sys.argv[1]
trials = [int(trial) for trial in trials_str.split(',')]

# if you want to plot starting from a certain epoch (for zooming purposes)
zoom = int(sys.argv[2])

legends = []
train_loss_plots = plt.subplot(211)
val_loss_plots = plt.subplot(212)

for trial in trials:

    # add this trial label to the legend
    legends.append("trial " + str(trial).zfill(3))
    
    trial_log_file = os.path.join("DeepNav_results", "trial_" + str(trial).zfill(3), "model_history_log.csv")
    # check if there is a log csv file for this trial and read it as a pandas frame
    try:
        df = pd.read_csv(trial_log_file)
    except:
        print("didn't find ", trial_log_file)
        continue
    
    # extract the training and validation losses vectors from the data frame
    df.head()
    df = df[["loss", "val_loss"]]
    train_loss = df.loss
    val_loss = df.val_loss
    
    # create epochs vector (x-axis)
    epochs = list(range(zoom, len(train_loss)))
    
    # print total epochs, minimum loss and val_loss along with their epochs
    print("trial:", trial, ", epochs:", val_loss.size, end=", ")
    print("min training loss:", f'{np.min(train_loss):.4f}', "at epoch", np.argmin(train_loss), end=", ") 
    print("min validation loss:", f'{np.min(val_loss):.4f}', "at epoch", np.argmin(val_loss),)

    train_loss_plots.plot(epochs, train_loss[zoom:])
    val_loss_plots.plot(epochs, val_loss[zoom:])
    

train_loss_plots.grid(True)
train_loss_plots.set_ylabel("Training Loss (MAE)")
train_loss_plots.legend(legends)

val_loss_plots.grid(True)
val_loss_plots.set_ylabel("Validation Loss (MAE)")
val_loss_plots.legend(legends)

plt.xlabel("epochs")
plt.savefig("losses.pdf")
plt.show()