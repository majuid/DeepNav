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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# trials you want to plot
trials = [1, 2]
legends = []

for trial in trials:
    
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
    
    loss = df.loss
    val_loss = df.val_loss
    
    # if you want to plot starting from a certain epoch (for zooming purposes)
    zoom = 3
    
    epoch = np.linspace(1, len(loss)-1, len(loss)) - zoom
    
    # print total epochs, minimum loss and val_loss along with their epochs
    print("trial:", trial, ", epochs:", val_loss.size, end=", ")
    print("min_loss:", f'{np.min(loss):.4f}', "at epoch", np.argmin(loss), end=", ") 
    print("min val_loss:", f'{np.min(val_loss):.4f}', "at epoch", np.argmin(val_loss),)


    plt.plot(epoch[zoom:], loss[zoom:])
    plt.plot(epoch[zoom:], val_loss[zoom:])
    plt.grid(True)
    
    # add this trial labels to the legend
    legends.append("trial " + str(trial).zfill(3) + " training loss")
    legends.append("trial " + str(trial).zfill(3) + " validation loss")



plt.legend(legends)
plt.xlabel("epochs")
plt.ylabel("Loss (MAE)")

plt.savefig("losses.pdf")

plt.show()
