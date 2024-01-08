import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from get_directories import get_data_dir, get_robot_maze_directory
from calculate_pos_and_dir import get_goal_coordinates

def plot_trial_path(trial_time, dlc_data):
    pass

def get_positional_occupancy(dlc_data):
    pass


def concatenate_dlc_data(dlc_data):
    for i, d in enumerate(dlc_data.keys()):

        # calculate frame intervals
        times = dlc_data[d]['ts'].values
        frame_intervals = np.diff(times)
        # one less interval than frames, so we'll just replicate the last interval
        frame_intervals = np.append(frame_intervals, frame_intervals[-1])

        # add frame intervals to dlc_data
        frame_intervals = frame_intervals/1000 # convert to seconds
        # round to 3 decimal places, i.e. 1 ms
        frame_intervals = np.round(frame_intervals, 3)
        dlc_data[d]['durations'] = frame_intervals



        if i==0:
            dlc_data_concat = dlc_data[d]
        
        else:
            dlc_data_concat = pd.concat([dlc_data_concat, dlc_data[d]], 
                    ignore_index=True)



        dlc_data[d] = pd.concat(dlc_data[d], ignore_index=True)
    pass

def get_directional_occupancy():
    pass

def get_rel2goal_occupancy():
    pass

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load dlc_data which has the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_final.pkl')
    with open(dlc_pickle_path, 'rb') as f:
        dlc_data = pickle.load(f)

    # load the platform coordinates, from which we can get the goal coordinates
    robot_maze_dir = get_robot_maze_directory()
    platform_path = os.path.join(robot_maze_dir, 'workstation',
            'map_files', 'platform_coordinates.pickle')
    with open(platform_path, 'rb') as f:
        platform_coordinates = pickle.load(f)

    # get goal coordinates 
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    # concatenate dlc_data
    dlc_data_concat = concatenate_dlc_data(dlc_data)


    # calculate positional occupancy
    positional_occupancy = get_positional_occupancy(dlc_data)
