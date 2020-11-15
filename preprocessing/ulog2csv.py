#! /usr/bin/env python

"""
Convert a ULog file into CSV file(s)
"""

from __future__ import print_function
import os
from ulog2csv_core import ULog

def convert_ulog2csv(ulog_file_name, messages, output, delimiter=',', disable_str_exceptions=False):
    """
    Coverts and ULog file to a CSV file.

    :param ulog_file_name: The ULog filename to open and read
    :param messages: A list of message names
    :param output: Output file path
    :param delimiter: CSV delimiter

    :return: None
    """

    msg_filter = messages.split(',') if messages else None

    ulog = ULog(ulog_file_name, msg_filter, disable_str_exceptions)
    data = ulog.data_list

    output_file_prefix = ulog_file_name[:-41]
    # strip '.ulg'
    if output_file_prefix.lower().endswith('.ulg'):
        output_file_prefix = output_file_prefix[:-4]

    # write to different output path?
    if output:
        base_name = os.path.basename(output_file_prefix)
        output_file_prefix = os.path.join(output, base_name)

    for d in data:
        fmt = '{0}_{1}_{2}.csv'
        output_file_name = fmt.format(output_file_prefix, d.name, d.multi_id)
        fmt = 'Writing {0} ({1} data points)'
        # print(fmt.format(output_file_name, len(d.data['timestamp'])))
        with open(output_file_name, 'w') as csvfile:

            # use same field order as in the log, except for the timestamp
            data_keys = [f.field_name for f in d.field_data]
            data_keys.remove('timestamp')
            data_keys.insert(0, 'timestamp')  # we want timestamp at first position

            # we don't use np.savetxt, because we have multiple arrays with
            # potentially different data types. However the following is quite
            # slow...

            # write the header
            csvfile.write(delimiter.join(data_keys) + '\n')

            # write the data
            last_elem = len(data_keys)-1
            for i in range(len(d.data['timestamp'])):
                for k in range(len(data_keys)):
                    csvfile.write(str(d.data[data_keys[k]][i]))
                    if k != last_elem:
                        csvfile.write(delimiter)
                csvfile.write('\n')

# change the working directory to the DeepNav_data directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))


ulg_dir =  os.path.join(os.path.pardir, "DeepNav_data", "ulg_files")
csvs_dir = os.path.join(os.path.pardir, "DeepNav_data", "flight_csvs")

if not os.path.isdir(csvs_dir) :
    os.mkdir(csvs_dir)

# Iterate on the flights (one folder per flight)
for flight_number, flight_name in enumerate(sorted(os.listdir(ulg_dir))):
    
    print("processing flight number ", flight_number, "\t", flight_name)

    output_folder = os.path.join(csvs_dir, flight_name[:-41])

    # if this ulg is already converted
    if os.path.isdir(output_folder) : 
        print("This log was processed earlier, skipping to the next!")
        continue
    
    os.mkdir(output_folder)

    messages = "sensor_combined,estimator_status,vehicle_gps_position,vehicle_air_data,vehicle_magnetometer"
    convert_ulog2csv(os.path.join(ulg_dir, flight_name), messages, output = output_folder)