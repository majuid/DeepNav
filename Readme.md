# Readme

## DeepNav Directory Structure

DeepNav
    
    |_ DeepNav_data : contains all data used for training
        |_ ulg_files: log files as downloaded from the database (.ulg)
        |_ flight_csvs: for every ulg file, there is a folder containing
                        multiple csv files, one csv for each message
                        at least there sould be
                            -estimator_status
                            -sensor_combined
                            -vehicle_air_data
                            -vehicle_magnetometer
        |_ combined_csvs: multiple csvs combined into one csv per flight
            |_ untrimmed
                |_ original
                |_ differenced: deltas using np.diff
            |_ flight_names.csv: contains names of all flights
                                in front of a name you can right the start and
                                end minutes you wish to keep the data in between
                                (trim outward) .. you can write "delete" to delete
                                the flight 
            |_ trimmed_csvs
                |_ original
                |_ differenced
                    |_training
                    |_validation
            |_ deleted_csvs
                |_ original
                |_ differenced
            