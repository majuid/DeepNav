#! /usr/bin/env python

"""
ulog2csv.py produces a directory containing different csv messages for each
ulg file. This script combines these csvs into two csvs per log, one is the combined
averaged data, and the other is a differenced version. It also creates a list of the
log files names and one PDF file containing the down positions of all flights
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_signal(dataset, signal_num, title, y_label="Down Position", signal_name=None):
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

def compress_sensor_data(sensor_timestamps, kf_timestamps, sensor_data):
    """
    compresses an array of measurements to be of the same length of the 
    kalman filter output (by averaging between each two KF steps)
    
    Arguments
        sensor_timestamps: numpy array of timestamps of the sensor measurements
        kf_timestamps: numpy array of timestamps of the EKF outputs
        sensor_data: numpy array of the sensor measurements (same length as sensor_timestamps)
    Return
        sensor_averaged: numpy array of sensor data of the same length of the KF outputs
    """

    # search for kf timestamps in the sensor timestamps
    sensor_indices = np.searchsorted(sensor_timestamps, kf_timestamps)
    
    # altimeter has only one colum
    if len(sensor_data.shape) < 2:
        n_colums = 1
    else:
        n_colums = sensor_data.shape[1]
    
    sensor_averaged = np.zeros(shape = (sensor_indices.size, n_colums))
    i = 0
    startIdx = 0

    # average all sensor data received between each two consecutive kf stamps    
    for endIdx in sensor_indices:
        sensor_averaged[i] = np.mean(sensor_data[startIdx:endIdx], axis = 0)
        startIdx = endIdx
        i = i + 1

    return sensor_averaged

def read_file(file_name, colums):
    """
    reads the first colum (timestamps) of a csv file into integer np array
    and other specified (data) colums into float np array 
    """
    sensor_timestamps = np.genfromtxt(file_name, delimiter=',', skip_header=1,  usecols=(0), dtype = np.int32)
    sensor_data = np.genfromtxt(file_name, delimiter=',', skip_header=1,  usecols=colums )
    
    return sensor_timestamps, sensor_data

# change the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

input_dir = os.path.join(os.path.pardir, "DeepNav_data", "flight_csvs")
output_dir = os.path.join(os.path.pardir, "DeepNav_data", "combined_csvs")
output_dir_orig = os.path.join(output_dir, "untrimmed", "original")
output_dir_diff = os.path.join(output_dir, "untrimmed", "differenced")

if not os.path.isdir(output_dir_orig) :
    os.makedirs(output_dir_orig)

if not os.path.isdir(output_dir_diff) :
    os.makedirs(output_dir_diff)

# create a csv containing the names of all the flights
# this will be used for manual inspection and cleaning
flight_names = sorted(os.listdir(input_dir))
flight_names_csv = os.path.join(output_dir, "flight_names.csv")
with open(flight_names_csv, "w") as f:
    writer = csv.writer(f)
    for flight_name in flight_names:
        writer.writerow([flight_name])

# iterate on the flights (one folder per flight)
total_flights_num = len(flight_names) - 1

# combine all the down position plots (of all flights) in a single pdf
all_flights_pdf = os.path.join(output_dir, "all_down_positions.pdf")
with PdfPages(all_flights_pdf) as pdf:

    for flight_number, flight_name in enumerate(flight_names):
        
        print("processing flight number", flight_number, "/", total_flights_num, "\t", flight_name)
        
        # skip logs processed in earlier run
        if os.path.isfile(os.path.join(output_dir_orig, flight_name + ".csv")) : 
            print("This log was processed earlier, skipping to the next!")
            continue

        # input files names
        base_name = os.path.join(input_dir, flight_name, flight_name)
        kf_file = base_name + '_estimator_status_0.csv'
        imu_file = base_name +  '_sensor_combined_0.csv'
        baro_file = base_name +  '_vehicle_air_data_0.csv'
        mag_file = base_name +  '_vehicle_magnetometer_0.csv'

        # read files
        kf_timestamps, kf_data = read_file(kf_file, colums = range(1,11)) # attitude, velocity & position components
        imu_timestamps, imu_data = read_file(imu_file, colums = [1,2,3,6,7,8]) # gyro & accel measurements
        baro_timestamps, baro_data = read_file(baro_file, colums = [1,2]) # altitude & temperature 
        mag_timestamps, mag_data = read_file(mag_file, colums = [1,2,3]) # 3 magnetic field coponents

        # trim the kf vector to start after every individual sensor
        kfStart = kf_timestamps[0]
        latest_starter = max([imu_timestamps[0], baro_timestamps[0], mag_timestamps[0]])
        kfStartIdx = 0
        if kfStart < latest_starter:
            kfStartIdx = np.argmax(kf_timestamps>latest_starter)
        kf_timestamps = kf_timestamps[kfStartIdx:]
        kf_data = kf_data[kfStartIdx:]

        # average all sensor data received between each two consecutive kf stamps
        imu_averaged = compress_sensor_data(imu_timestamps, kf_timestamps, imu_data)
        baro_averaged = compress_sensor_data(baro_timestamps, kf_timestamps, baro_data)
        mag_averaged = compress_sensor_data(mag_timestamps, kf_timestamps, mag_data)
        
        # combine all the averaged data in one array (11+10) x n
        dataset_original = np.concatenate((imu_averaged, mag_averaged, baro_averaged, kf_data), axis=1)
        header_orig = "p,q,r,a_x,a_y,a_z,m_x,m_y,m_z,h,T,q0,q1,q2,q3,Vn,Ve,Vd,Pn,Pe,Pd"
        header_diff = "p,q,r,a_x,a_y,a_z,m_x,m_y,m_z,dh,dT,dq0,dq1,dq2,dq3,dVn,dVe,dVd,dPn,dPe,dPd"

        # remove rows that contain nans
        dataset_original = dataset_original[~np.isnan(dataset_original).any(axis=1)]

        # remove the ground time in the beggining and end of flight
        takeoff_height = -1
        land_height = -1
        down_position = dataset_original[:,-1]
        takeoff_index = np.argmax(down_position<takeoff_height)
        land_index = -1 - np.argmax(down_position[::-1]<land_height)
        dataset_original = dataset_original[takeoff_index:land_index, :]

        # create a differenced copy of all the data (deltas instead of absolute values)
        dataset_differenced = np.diff(dataset_original, axis=0)

        # Use original features along with differenced labels
        dataset_half_differneced = np.hstack([dataset_original[1:, 0:9], dataset_differenced[:, 9:]])

        # output files
        output_file_orig = os.path.join(output_dir_orig, flight_name + ".csv")
        output_file_diff = os.path.join(output_dir_diff, flight_name + ".csv")
        
        # save the datasets
        np.savetxt(output_file_orig, dataset_original, delimiter=",", header=header_orig)
        np.savetxt(output_file_diff, dataset_half_differneced, delimiter=",", header=header_diff)

        # save the down position plot, used for manual inspection of logs
        plot_signal(dataset_original, signal_num=-1, y_label="Down Position (meters)", title=flight_name)