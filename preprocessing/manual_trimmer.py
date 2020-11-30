#! /usr/bin/env python

"""
This script allows you to manually trim the ground time before takeoff
and after landing. It also allows the deletion of bad logs.
To use this script, inspect "all_down_positions.pdf" produced by combine_csvs.py
and in "flight_names.csv" define a new start and end time, or write delete
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_signal(dataset, signal_num, title, y_label="Down Position"):
    """
    Arguments
        dataset: 2D np array, colums are different signals, rows are timesteps
        signal_num: required colum to plot
        signal_name: used in legend, to plot multiple signals in one plot
    """
    series = dataset[:,signal_num]
    time_steps = np.linspace(0, series.size-1, series.size) * 0.2 / 60
    plt.plot(time_steps, series)
    plt.grid(True)
    plt.title(title)
    plt.xlabel("time (minutes)")
    plt.ylabel(y_label)
    pdf.savefig()
    plt.close()

# change the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# input and output directories
input_root_dir = os.path.join(os.path.pardir, "DeepNav_data", "combined_csvs")
input_csvs_dir = os.path.join(input_root_dir, "untrimmed")
deleted_csvs_dir = os.path.join(input_root_dir, "deleted")
trimmed_csvs_dir = os.path.join(input_root_dir, "trimmed")

if not os.path.isdir(deleted_csvs_dir): 
    os.makedirs(deleted_csvs_dir)
if not os.path.isdir(trimmed_csvs_dir): 
    os.makedirs(trimmed_csvs_dir)

# read the names of the logs from txt (created manually), along with
# the required start and end times
flight_names_csv = os.path.join(input_root_dir, "flight_names.csv")
with open(flight_names_csv, 'r') as f:
    logs_lines = f.readlines()

# a list to store the bad logs that will be deleted
deleted_list = []

total_flights_num = len(logs_lines)

# to create multi page pdf plot
pdf_name = os.path.join(input_root_dir, "trimmed_down_positions.pdf")
with PdfPages(pdf_name) as pdf:

    for flight_num, bad_logs_line in enumerate(logs_lines):
        
        # if this log will be edited, edited = True
        edited = False

        # read the log name, manually determined start and end points for trimming
        log_name, start, end = bad_logs_line.strip().split(",")
        flight_name = log_name + ".csv"

        print("trimming", flight_num, "/", total_flights_num)

        # read this log's csv
        try:
            flight_data = np.genfromtxt(os.path.join(input_csvs_dir, flight_name), delimiter=',', skip_header=1)
        except:
            # csv not found
            continue
        
        # if start or end values are not given, don't trim that side
        # and if start is "delete", the entire log will be deleted
        if start == "":
            start = None
        elif start == "delete":
            deleted_list.append(log_name)
            os.rename(os.path.join(input_csvs_dir, flight_name), os.path.join(deleted_csvs_dir, flight_name))
            continue
        else:
            start = int(float(start) * 60 * 5)
            edited = True

        if end == "":
            end = None
        else:
            end = int(float(end) * 60 * 5)
            edited = True

        # trim
        flight_data = flight_data[start:end, :]

        # output files names
        output_file = os.path.join(trimmed_csvs_dir, flight_name)

        # save the trimmed files
        header = "w_x,w_y,w_z,a_x,a_y,a_z,m_x,m_y,m_z,h,T,q0,q1,q2,q3,Vn,Ve,Vd,Pn,Pe,Pd"
        np.savetxt(output_file, flight_data, delimiter=",", header=header, comments='')
        
        # save the down position plot, to compare it with the originals
        plot_signal(flight_data, signal_num=-1, y_label="Down Position (meters)", title=log_name)

# write the names of the deleted files to a txt
deleted_flight_names_csv = os.path.join(input_root_dir, "deleted_logs.csv")
with open(deleted_flight_names_csv, "w") as outfile:
    outfile.write("\n".join(deleted_list))