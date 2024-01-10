import os
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate

from get_directories import get_data_dir, get_robot_maze_directory


def interpolate_rads(og_times, og_rads, target_times):
    unwrapped_ogs = np.unwrap(og_rads)
    f_rad = interpolate.interp1d(og_times, unwrapped_ogs)
    target_rads_unwrapped = f_rad(target_times)
    target_rads = (target_rads_unwrapped + np.pi) % (2 * np.pi) - np.pi
    
    return target_rads


def get_unit_position_and_directions(dlc_data, unit):

    for t in unit.keys():
        # get the spike times for the unit
        spike_times = unit[t]
        # make spike times a data frame with samples as the column name
        spike_times = pd.DataFrame({'samples': spike_times})

        # get the position data for the trial
        pos_data = dlc_data[t]['x']

        # create the interpolation functions
        f_x = interpolate.interp1d(dlc_data[t]['video_samples'], dlc_data[t]['x'])
        f_y = interpolate.interp1d(dlc_data[t]['video_samples'], dlc_data[t]['y'])

        spike_times['x'] = f_x(spike_times)
        spike_times['y'] = f_y(spike_times)

        # interpolate the directional data
        directional_data = ['hd']
        dlc_columns = dlc_data[t].columns
        for c in dlc_columns:
            if 'relative_direction' in c:
                directional_data.append(c)




    return unit

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    restricted_units_file = os.path.join(spike_dir, 'restricted_units.pickle')
    with open(restricted_units_file, 'rb') as handle:
        restricted_units = pickle.load(handle)

    # load neuron classification data
    neuron_types_file = os.path.join(spike_dir, 'average_waveforms', 
                                     'neuron_types.pickle')
    with open(neuron_types_file, 'rb') as handle:
        neuron_types = pickle.load(handle)    

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_final.pkl')
    with open(dlc_pickle_path, 'rb') as f:
        dlc_data = pickle.load(f)

    # loop through units and calculate positions and various directional correlates
    for unit in restricted_units.keys():
        restricted_units[unit] = get_unit_position_and_directions(dlc_data, restricted_units[unit])

    pass