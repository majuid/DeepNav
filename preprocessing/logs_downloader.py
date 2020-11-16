#! /usr/bin/env python3

""" Script to download public logs """

import os
import json
import datetime
import sys
import time
import yaml
import requests

# configuration tables (map modes and errors to IDs)

flight_modes_table = {
    0: ('Manual', '#cc0000'), # red
    1: ('Altitude', '#eecc00'), # yellow
    2: ('Position', '#00cc33'), # green
    10: ('Acro', '#66cc00'), # olive
    14: ('Offboard', '#00cccc'), # light blue
    15: ('Stabilized', '#0033cc'), # dark blue
    16: ('Rattitude', '#ee9900'), # orange

    # all AUTO-modes use the same color
    3: ('Mission', '#6600cc'), # purple
    4: ('Loiter', '#6600cc'), # purple
    5: ('Return to Land', '#6600cc'), # purple
    6: ('RC Recovery', '#6600cc'), # purple
    7: ('Return to groundstation', '#6600cc'), # purple
    8: ('Land (engine fail)', '#6600cc'), # purple
    9: ('Land (GPS fail)', '#6600cc'), # purple
    12: ('Descend', '#6600cc'), # purple
    13: ('Terminate', '#6600cc'), # purple
    17: ('Takeoff', '#6600cc'), # purple
    18: ('Land', '#6600cc'), # purple
    19: ('Follow Target', '#6600cc'), # purple
    20: ('Precision Land', '#6600cc'), # purple
    21: ('Orbit', '#6600cc'), # purple
    }

vtol_modes_table = {
    1: ('Transition', '#cc0000'), # red
    2: ('Fixed-Wing', '#eecc00'), # yellow
    3: ('Multicopter', '#0033cc'), # dark blue
    }

error_labels_table = {
    # the labels (values) have to be capitalized!
    # 'validate_error_labels_and_get_ids' will return an error otherwise
    1: 'Other',
    2: 'Vibration',
    3: 'Airframe-design',
    4: 'Sensor-error',
    5: 'Component-failure',
    6: 'Software',
    7: 'Human-error',
    8: 'External-conditions'
       # Note: when adding new labels, always increase the id, never re-use a lower value
    }

def flight_modes_to_ids(flight_modes):
    """
    returns a list of mode ids for a list of mode labels
    """
    flight_ids = []
    for i in flight_modes_table:
        if flight_modes_table[i][0] in flight_modes:
            flight_ids.append(i)
    return flight_ids


def error_labels_to_ids(error_labels):
    """
    returns a list of error ids for a list of error labels
    """
    error_id_table = {label: id for id, label in error_labels_table.items()}
    error_ids = [error_id_table[error_label] for error_label in error_labels]
    return error_ids


""" main script entry point """

# change the working directory to be the script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# read the user arguments (including filters for log download)
with open("downloader_options.yaml", 'r') as stream:
    arguments = yaml.safe_load(stream)

# create a folder to store database info files
db_info_dir = os.path.join(os.path.pardir, "DeepNav_data", "database_info_files")
if not os.path.isdir(db_info_dir):
    os.makedirs(db_info_dir)

# if the user has a database info file, use it
db_entries_list = []
if arguments["use_local_db_info"]:
    db_file = arguments["local_db_info_file"]
    db_path = os.path.join(db_info_dir, db_file)
    if os.path.isfile(db_path):
        with open(db_path, 'r') as fin:
            db_entries_list = json.load(fin)
    else:
        print("didn't find the database info file")
        exit()

# or download the updated database info and save it
else:
    db_file = "px4_db_info_" + datetime.datetime.now().strftime("%d%b%Y") + ".json"
    db_path = os.path.join(db_info_dir, db_file)
    print("attempting to download database info file ..")
    try:
        # the db_info_api sends a json file with a list of all public database entries
        db_entries_list = requests.get(url=arguments["db_api_info"]).json()
        print("downloaded database info, saving ...")
        with open(db_path, 'w') as fout:
            json.dump(db_entries_list, fout, indent=4)
        print("saved downloaded database")
    except:
        print("Server request failed, retry later")
        sys.exit()

print("\ndatabase info retreived!, number of logs : ", len(db_entries_list))

# find already existing logs in download folder (a log id is 36 characters)
download_folder = os.path.join(os.path.pardir, "DeepNav_data", "ulg_files")
if os.path.isdir(download_folder):
    lognames = sorted(os.listdir(download_folder))
    logfiles = [log_name[-40:-4] for log_name in lognames]
    existing_logs = frozenset(logfiles)
    print("found", len(existing_logs), "logs in the download folder")
else:
    print("no logs folder found, just created")
    existing_logs = frozenset([])
    os.makedirs(download_folder)

# filter logs for flight duration
if arguments["duration_max_m"] is not None and arguments["duration_min_m"] is not None:
    db_entries_list = [
        entry for entry in db_entries_list if entry['duration_s'] > arguments["duration_min_m"] * 60 and
         entry['duration_s'] < arguments["duration_max_m"] * 60]

    print("\nfiltered flight duration, number of remaining logs:", len(db_entries_list))

# apply other filters defined in downloader_options.yaml -> filters (including mav_type, flight_modes ... etc)
for filter_name, filter_values in arguments["filters"].items():
    if filter_values is not None:
        if filter_name == "flight_modes":
            # use modes ids
            filter_values = flight_modes_to_ids(filter_values)
            # keep any log that contains any of the given modes
            db_entries_list = [entry for entry in db_entries_list
                            if (set(entry[filter_name]) & set(filter_values))]
        elif filter_name == "num_logged_errors":
            # only keep logs with the required number of errors (zero is recommended)
            db_entries_list = [entry for entry in db_entries_list
                            if entry[filter_name] == filter_values]
        else:
            # for any other filter keep any log that satisfies any filter value
            # for example, if multiple hardware allowed, keep logs that use any of them
            filter_values = [value.lower() for value in filter_values]
            db_entries_list = [entry for entry in db_entries_list
                            if entry[filter_name].lower() in filter_values]

        print("filtered for", filter_name, ",  number of remaining logs:", len(db_entries_list))

# remove simulated logs from qground station
db_entries_list = [entry for entry in db_entries_list if entry["source"] != "QGroundControl"]
print("removed simulated logs,  number of remaining logs: ", len(db_entries_list))
print("")     

# sort list to first download the shortest logs
db_entries_list = sorted(db_entries_list, key=lambda x: x['duration_s'])

# save the filtered db file
filtered_db_file = "filtered_" + db_file
filtered_db_path = os.path.join(db_info_dir, filtered_db_file)
with open(filtered_db_path, 'w') as fout:
    json.dump(db_entries_list, fout, indent=4)
    print("saved filtered db\n")

# download the log files if required
if not arguments['save_db_info_only']:

    # set number of files to download
    max_num_logs = len(db_entries_list)
    if arguments["max_num"] > 0:
        max_num_logs = min(max_num_logs, arguments["max_num"])
    n_downloaded = 0
    n_skipped = 0

    # loop on the db info list (logs)
    for log_num, log in enumerate(db_entries_list[:max_num_logs]):
        entry_id = log['log_id']
        log_duration = log['duration_s'] / 60
        log_name = str(log_num).zfill(4) + '_' + f'{log_duration:.2f}' + '_' + entry_id + ".ulg"

        num_tries = 0
        for num_tries in range(100):
            try:
                if arguments["overwrite"] or entry_id not in existing_logs:

                    file_path = os.path.join(download_folder, log_name)

                    print('downloading {:}/{:} ({:})'.format(log_num + 1, max_num_logs, log_name))
                    request = requests.get(url=arguments["download_api"] + "?log=" + entry_id, stream=True)
                    with open(file_path, 'wb') as log_file:
                        for chunk in request.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                log_file.write(chunk)
                    n_downloaded += 1
                
                else:
                    print("skipping", log_name, ", already existing")
                    n_skipped += 1

                break
            except Exception as ex:
                print(ex)
                print('Waiting for 30 seconds to retry')
                time.sleep(30)
        if num_tries == 99:
            print('Retried', str(num_tries + 1), 'times without success, exiting.')
            sys.exit(1)


    print('{:} logs downloaded to {:}, skipped {:}'.format(n_downloaded, download_folder, n_skipped))