import os
import numpy as np
import pandas as pd
import pickle

from get_directories import get_data_dir 
from process_dlc_data import load_dlc_processed_pickle

def restrict_times_to_interval(times_to_restrict, interval):

    restricted_times = times_to_restrict[(times_to_restrict >= 
                        interval[0]) & (times_to_restrict <= interval[1])]
    return restricted_times


def restrict_spikes_to_trials(units, dlc_data):
    
    restricted_units = {}

    for u in units.keys():
        # get the spike times for the unit
        spike_times = units[u]['samples']

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


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    
    # load the dlc data, which contains the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_final.pkl')
    dlc_data = load_dlc_processed_pickle(dlc_pickle_path)

    
    # load the spike data
    unit_dir = os.path.join(data_dir, 'spike_sorting')
    units_file = os.path.join(unit_dir, 'unit_spike_times.pickle')
    with open(units_file, 'rb') as handle:
        units = pickle.load(handle)
    
    restricted_units = restrict_spikes_to_trials(units, dlc_data)
    restricted_units_file = os.path.join(unit_dir, 'restricted_units.pickle')
    with open(restricted_units_file, 'wb') as handle:
        pickle.dump(restricted_units, handle, protocol=pickle.HIGHEST_PROTOCOL)

    del restricted_units

    with open(restricted_units_file, 'rb') as handle:
        restricted_units = pickle.load(handle)

    pass