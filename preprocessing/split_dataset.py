#! /usr/bin/env python

"""
split the dataset folder into train and test folders
"""

import os
import random

os.chdir(os.path.dirname(os.path.abspath(__file__)))

csvs_dir = os.path.join(os.pardir, "DeepNav_data", "combined_csvs", "trimmed")
csvs_list = sorted(os.listdir(csvs_dir))

random.shuffle(csvs_list)

train_folder = os.path.join(csvs_dir, "training")
os.mkdir(train_folder)
valid_folder = os.path.join(csvs_dir, "validation")
os.mkdir(valid_folder)

train_valid_ratio = 0.85
split = int(len(csvs_list) * train_valid_ratio)

for log in csvs_list[:split]:
    os.rename(os.path.join(csvs_dir, log), os.path.join(train_folder, log))

for log in csvs_list[split:]:
    os.rename(os.path.join(csvs_dir, log), os.path.join(valid_folder, log))

