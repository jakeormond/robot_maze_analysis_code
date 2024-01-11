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
        
        # make spike times a data frame with samples as the column name
        unit[t] = pd.DataFrame({'samples': unit[t]})

        # create the interpolation functions
        f_x = interpolate.interp1d(dlc_data[t]['video_samples'], dlc_data[t]['x'])
        f_y = interpolate.interp1d(dlc_data[t]['video_samples'], dlc_data[t]['y'])

        unit[t]['x'] = np.round(f_x(unit[t]['samples']), 2)
        unit[t]['y'] = np.round(f_y(unit[t]['samples']), 2)

        # interpolate the directional data
        directional_data_cols = ['hd']
        dlc_columns = dlc_data[t].columns
        for c in dlc_columns:
            if 'relative_direction' in c:
                directional_data_cols.append(c)

        for c in directional_data_cols:
            
            target_rads = interpolate_rads(dlc_data[t]['video_samples'], dlc_data[t][c], 
                                       unit[t]['samples'])
            unit[t][c] = np.around(target_rads, 3)

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

    # save the restricted units
    restricted_units_file = os.path.join(spike_dir, 'units_w_behav_correlates.pickle')
    with open(restricted_units_file, 'wb') as handle:
        pickle.dump(restricted_units, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pass