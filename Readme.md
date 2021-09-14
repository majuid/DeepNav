# DeepNav | An Intelligent Inertial Navigation System
This is a neural network that predicts a drone's
local position from the raw IMU, Barometer, and Magnetometer measurements. No GPS, camera or any other sensor is needed. This system works with the [PX4 autopilot](https://px4.io/) using [MAVROS](http://wiki.ros.org/mavros) in realtime.

## Reference
If you need to dive deeper into the science behind DeepNav, you can download the paper preprint 

https://arxiv.org/abs/2109.04861

please cite as

    @misc{abdulmajuid2021gpsdenied,
          title={GPS-Denied Navigation Using Low-Cost Inertial Sensors and Recurrent Neural Networks}, 
          author={Ahmed AbdulMajuid and Osama Mohamady and Mohannad Draz and Gamal El-bayoumi},
          year={2021},
          eprint={2109.04861},
          archivePrefix={arXiv},
          primaryClass={eess.SP}
    }

## Considerations

- This is a beta release, to avoid crashes, don't use it for flight control as an alternative to the tradional [EKF/GPS](https://docs.px4.io/master/en/advanced_config/tuning_the_ecl_ekf.html). Instead, you are welcome to test, enhance or compare it to other estimators.

- Good results were obtained on quadrotors running in  [auto modes](https://docs.px4.io/master/en/flight_modes/) and using [Pixhawk4](https://docs.px4.io/master/en/flight_controller/pixhawk4.html). Other controllers, frames, and modes will probably result in a reduced accuracy.

- The system was tested on Ubuntu 18.04 and Ubuntu 20.04, both with TensorFlow 2.2.0

- You don't need a GPU to *run* DeepNav, you only need it to *train* it.

## Basic Usage: using the trained network

If you don't want to apply any modifications and use the trained network as it is, follow these steps. Otherwise, go to "Advanced Usage" below.

### Requirements
- To test in a real flight, you need a ready-to-fly drone with Pixhawk4 running PX4 (or you can just test on a saved log)

- A [companion computer](https://docs.px4.io/master/en/companion_computer/pixhawk_companion.html) to run the network, you can use the ground station laptop instead.

- [TensorFlow 2.2.0](https://www.tensorflow.org/ "$ conda install tensorflow=2.2.0") and [MAVROS](https://docs.px4.io/master/en/ros/mavros_installation.html).

- A realtime visualization tool to compare the results to the ground truth. I recommend [PlotJuggler](https://github.com/facontidavide/PlotJuggler "$ sudo apt install ros-<distro>-plotjuggler-ros"). 


### Try the provided log sample

1. `$ git clone https://github.com/majuid/DeepNav.git`

1. Copy the directory `/DeepNav/msg` to your `mavros` catking workspace, ex. `~/catkin_ws/src/mavros/mavros`  and edit [these](http://wiki.ros.org/ROS/Tutorials/CreatingMsgAndSrv) necessary files then build your `mavros` workspace

1. Download the trained network [here](https://drive.google.com/file/d/1aD84q2ZBdiBsw_gP0yTfSXvEAu1rSqex/view?usp=sharing) and extract it to `/DeepNav/DeepNav_results/`

1. Download the sample log [here](https://drive.google.com/file/d/1YZJ8ty6Zw7g0bgq3ZqYWtrH9wQ-rFHhu/view?usp=sharing) and extract it to `/DeepNav/DeepNav_data/`

1. Source your catkin workspace, ex.

    `$ source ~/catkin_ws/devel/setup.bash`


    and start ros

    `$ roscore`

1. Open a new terminal tab (Ctrl+Shift+T in gnome) and activate the conda environment where you installed tensorflow (if you made one) then source the catkin workspace again

1. Start the network

    `$ cd ../realtime_inference` 

    `$ ./run_inference.py 6 100`

    note: `6` is the trial folder in which the tf_model is saved and `100` is the `window_size` for that model

1. Wait until this message is displayed:

    `you can start publishing mavros sensor messages now`

    then repeat step 5, then
    
    `$ ./replay_log_csv.py 0057_5.12`

    note: `0057_5.12`  is the log folder inside  `/DeepNav_data/flight_csvs`

1. When this message is displayed:

    `mavros messages are now being published`
    
    the network will start creating a window from the recieved sensor measurements, and messages like this will be displayed in the `run_inference.py` terminal tab: 
    
    `window size:  1 counters: {'imu': 6, 'mag': 2, 'alt': 2}`
    
1. When this window becomes equal to the model's `window_size`, inferenece will start, and this message will be displayed in `run_inference.py` tab:

    `*` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `first prediction`  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  `*`

    then messages like this will appear alternating with the `window_size` message:

    `inference_time :  0.1191554230026668`

1. Now the network predictions are being published to `/deep_nav/nn_predictions` ros topic

    you can visualize the predictions by running

    `$ rosrun plotjuggler plotjuggler -n -l PlotJuggler_layout.xml`

#### Expected Result - Video

[![DeepNav Demo](http://img.youtube.com/vi/MtzwcpFkFA0/0.jpg)](https://www.youtube.com/watch?v=MtzwcpFkFA0 "DeepNav | An Intelligent Inertial Navigation System Demo")


### Replay another log

all the steps from the previous section will be performed, except the sample log download step will be replaced by the following

1. Download a log from the [PX4 Flight Review Database](https://review.px4.io/browse) and place it in `/DeepNav/DeepNav_data/ulg_files`. You can also use your own log. For convenience, give the log name a prefix, for example,  `c07ba992-c8a4-4f1f-a7ab-bd4661680a6d.ulg` becomes  `0001_c07ba992-c8a4-4f1f-a7ab-bd4661680a6d.ulg`, the long hash will be stripped later.

1. Convert the binary ulg file into multiple csv files 

    `$ cd /DeepNav/preprocessing`

    `$ ./ulog2csv.py`

    This creates a folder with the name of this log's prefix in the `/DeepNav_data/flight_csvs` directory. You will supply the name of this folder to `replay_log_csv.py`


notes: 
- A new version of the PX4 logger saves the EKF states in an `.._estimator_states.csv` file instead of `.._estimator_status.csv` you'll need to modify its name in `/flight_csvs` yourself to proceed smoothly (i will account for this in a future update)

- The network is trained to make inferences at 5 Hz, so if the `estimator_status` file has the ground truth EKF values at a different rate, this might result in the prediction and truth plots in the visualizer to have issues, this won't affect the prediction quality itself, only the visualization.


## Advanced Usage: training a new network


1. Edit `/DeepNav/preprocessing/downloader_options.yaml` to define criteria for the logs to be downloaded from the flight review database

1. Start the download, this might download hundreds of gigabytes to your machine, depending on the filters you set in the previous step

    `$ cd preprocessing`

    `$ ./logs_downloader.py`

1. Convert all the logs to csv files

    `$ ./ulog2csv.py`

1. Combine the csv files of each log into a single csv

    `$ ./combine_csvs.py`

    this creates `all_down_positions.pdf` in `/DeepNav/DeepNav_data` directory, you can manually inspect the down positions to determine the duplicate or corrupted logs. You can then type `delete` in front of their names in `flight_names.csv` in the same directory. You can also type a start time and/or end time (in minutes) in the second and third columns in front of any flight name to trim its ends. 

1. Run the trimmer based on your manual inspection

    `$ ./manual_trimmer.py`

1. Split the dataset into training and validation sets (85% : 15%)

    `$ ./split_dataset.py`

1. Now you can modify the network hyperparameters in `DeepNav.py` and run a new training session

    `$ ./DeepNav.py`

    The current parameters in the script are far from ideal. I put values to allow the script to run quickly for you to make sure that postprocessing works fines. If the script exits without errors (like a missing dependancy), you can uncomment the tuned architecture and hyperparameters. A training session takes about 5 hours on Nvidia RTX 2070 Super.

Running `DeepNav.py` will create a `/trial_###` directory in the `/DeepNav/DeepNav_results` directory

This created directory will contain the trained model in `TF_SavedModel` and `Keras` formats, training and validation sets individual flights results, training and validation summaries, weights checkpoints (if you want to use early stopping), and loss history.
