#! /usr/bin/env python

"""
ulog2csv.py produces a directory containing different csv messages for each
ulg file. This script combines these csvs into one csv per log, containing the combined
averaged data with as many timesteps as the ekf. It also creates a list of the
log files names and one PDF file containing the down positions of all flights
"""

import os
import csv
import numpy as np
import pandas as pd
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

def compress_sensor_data(kf_timestamps, sensor_msg):
    """
    compresses an array of measurements to be of the same length of the 
    kalman filter output (by averaging between each two KF steps)
    
    Arguments
        kf_timestamps: numpy array of timestamps of the EKF outputs
        sensor_msg: a dictionary that has two np arrays:
                    timestamps: timestamps of the sensor measurements
                    data: sensor measurements (same length as sensor_timestamps)
    Return
        sensor_averaged: numpy array of sensor data of the same length of the KF outputs
    """

    # search for kf timestamps in the sensor timestamps
    sensor_indices = np.searchsorted(sensor_msg["timestamps"], kf_timestamps)   
    
    # average all sensor data received between each two consecutive kf stamps    
    startIdx = 0
    i = 0
    sensor_averaged = np.zeros((sensor_indices.size, sensor_msg["data"].shape[1]))
    for endIdx in sensor_indices:
        sensor_averaged[i] = np.mean(sensor_msg["data"][startIdx:endIdx], axis = 0)
        startIdx = endIdx
        i += 1

    return np.array(sensor_averaged)

# change the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(os.path.pardir, "DeepNav_data", "flight_csvs")
output_root_dir = os.path.join(os.path.pardir, "DeepNav_data", "combined_csvs")
output_csvs_dir = os.path.join(output_root_dir, "untrimmed")

if not os.path.isdir(output_csvs_dir) :
    os.makedirs(output_csvs_dir)

# create a csv containing the names of all the flights
# this will be used for manual inspection and cleaning
flight_names = sorted(os.listdir(input_dir))
flight_names_csv = os.path.join(output_root_dir, "flight_names.csv")
with open(flight_names_csv, "w") as f:
    writer = csv.writer(f)
    for flight_name in flight_names:
        writer.writerow([flight_name])

messages = {"ekf" : {"file" : '_estimator_status',     "cols" : ["states["+str(i)+"]" for i in range(10)]},
            "mag" : {"file" : '_vehicle_magnetometer', "cols" : ["magnetometer_ga["+str(i)+"]" for i in range(3)]},
            "baro": {"file" : '_vehicle_air_data',     "cols" : ["baro_alt_meter", "baro_temp_celcius"]},
            "imu" : {"file" : '_sensor_combined',      "cols" : ["gyro_rad["+str(i)+"]" for i in range(3)] + \
                                                                ["accelerometer_m_s2["+str(i)+"]" for i in range(3)]},
            }

# iterate on the flights (one folder per flight)
total_flights_num = len(flight_names) - 1

# combine all the down position plots (of all flights) in a single pdf
all_flights_pdf = os.path.join(output_root_dir, "all_down_positions.pdf")
with PdfPages(all_flights_pdf) as pdf:

    for flight_number, flight_name in enumerate(flight_names):
        
        print("combining csvs for flight number", flight_number, "/", total_flights_num, "\t", flight_name)
        
        # skip logs processed in earlier runs
        if os.path.isfile(os.path.join(output_csvs_dir, flight_name + ".csv")) : 
            print("This log was processed earlier, skipping to the next!")
            continue
        
        base_name = os.path.join(input_dir, flight_name, flight_name)
        
        log_is_corrupted = False
        
        for message in messages.values():
            file_name = base_name + message["file"] + "_0.csv"
            if os.path.isfile(file_name):
                message["timestamps"] = pd.read_csv(file_name, usecols=["timestamp"]).to_numpy().squeeze()
                message["data"] = pd.read_csv(file_name, usecols=message["cols"])[message["cols"]].to_numpy()
            else:
                log_is_corrupted = True
                break

        if log_is_corrupted:
            print("flight number", flight_number, "has a missing csv file, skipping!")
            continue

        # trim the kf vector to start after every individual sensor
        ekf_timestamps = messages["ekf"]["timestamps"]
        kfStart = ekf_timestamps[0]
        latest_starter = max([messages["imu"]["timestamps"][0], messages["baro"]["timestamps"][0],
                              messages["mag"]["timestamps"][0]])
        kfStartIdx = 0
        if kfStart < latest_starter:
            kfStartIdx = np.argmax(ekf_timestamps>latest_starter)
        ekf_timestamps = ekf_timestamps[kfStartIdx:]
        messages["ekf"]["data"] = messages["ekf"]["data"][kfStartIdx:]

        # average all sensor data received between each two consecutive kf stamps
        for msg_key, message in messages.items():
            if msg_key != "ekf":
                message["data"] = compress_sensor_data(ekf_timestamps, message)
        
        # combine all the averaged data in one array (11+10) x n
        combined_data = np.concatenate((messages["imu"]["data"], messages["mag"]["data"],
                                        messages["baro"]["data"], messages["ekf"]["data"]), axis=1)
        header = "w_x,w_y,w_z,a_x,a_y,a_z,m_x,m_y,m_z,h,T,q0,q1,q2,q3,Vn,Ve,Vd,Pn,Pe,Pd"

        # remove rows that contain nans
        combined_data = combined_data[~np.isnan(combined_data).any(axis=1)]

        # remove the ground time in before takeoff and after landing
        takeoff_height = -1
        land_height = -1
        down_position = combined_data[:,-1]
        takeoff_index = np.argmax(down_position<takeoff_height)
        land_index = -1 - np.argmax(down_position[::-1]<land_height)
        combined_data = combined_data[takeoff_index:land_index, :]

        # output file
        output_file = os.path.join(output_csvs_dir, flight_name + ".csv")
        
        # save the datasets
        np.savetxt(output_file, combined_data, delimiter=",", header=header, comments='')

        # save the down position plot, used for manual inspection of logs
        plot_signal(combined_data, signal_num=-1, y_label="Down Position (meters)", title=flight_name)