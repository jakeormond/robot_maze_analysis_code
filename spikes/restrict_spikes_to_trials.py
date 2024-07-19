import os
import numpy as np
import pandas as pd
import pickle

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir 
from position.process_dlc_data import load_dlc_processed_pickle
from utilities.load_and_save_data import save_pickle, load_pickle

def restrict_times_to_interval(times_to_restrict, interval):

    restricted_times = times_to_restrict[(times_to_restrict >= 
                        interval[0]) & (times_to_restrict <= interval[1])]
    return restricted_times


def restrict_spikes_to_trials(units, dlc_data):
    
    restricted_units = {}

    for u in units.keys():
        # get the spike times for the unit

        spike_times = units[u]
        if not isinstance(spike_times, np.ndarray): 
            spike_times = spike_times['samples']

        restricted_units[u] = {}

        for t in dlc_data.keys():
            # get the start and end times of the trial
            start_time = dlc_data[t]['video_samples'].iloc[0]
            end_time = dlc_data[t]['video_samples'].iloc[-1]

            # get the spike times that are in the trial
            spike_times_in_trial = restrict_times_to_interval(spike_times, 
                                                [start_time, end_time])

            # put spike times in restricted_units dictionary
            restricted_units[u][t] = spike_times_in_trial

    return restricted_units


def concatenate_unit_across_trials(unit):
    # unit is a dictionary with keys as trial numbers and values as dataframes
    
    for i, t in enumerate(unit.keys()):
        if i == 0:
            concatenated_unit = unit[t]
        else:
            # combine dataframes
            concatenated_unit = pd.concat([concatenated_unit, unit[t]], axis=0)

    return concatenated_unit


if __name__ == "__main__":
    
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    # data_dir = 'D:/analysis/og_honeycomb/rat7/6-12-2019'
    # data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network/og_honeycomb/rat7/6-12-2019'
    
    # load the dlc data, which contains the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

       
    # load the spike data
    unit_dir = os.path.join(data_dir, 'spike_sorting')

    # unit_dir = os.path.join(data_dir, 'physiology_data')
    units = load_pickle('unit_spike_times', unit_dir)
    # units = units['pyramid']
    
    # restricted_units = restrict_spikes_to_trials(units, dlc_data['hComb'])
    restricted_units = restrict_spikes_to_trials(units, dlc_data)
    save_pickle(restricted_units, 'restricted_units', unit_dir)

    pass