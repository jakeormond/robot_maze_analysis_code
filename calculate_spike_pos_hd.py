import os
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate

from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle


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

def bin_spikes_by_position(units, positional_occupancy):

    # get the x and y bins
    x_bins = positional_occupancy['x_bins']
    x_bins[-1] = x_bins[-1] + 1
    y_bins = positional_occupancy['y_bins']
    y_bins[-1] = y_bins[-1] + 1

    # loop through the units
    spike_rates_by_position = {}
    for u in units.keys():
        
        # loop through the trials getting all the spike positions
        for i, t in enumerate(units[u].keys()):
            # get the x and y positions
            if i == 0:
                x = units[u][t]['x']
                y = units[u][t]['y']
                samples = units[u][t]['samples']
            else:
                x = np.concatenate((x, units[u][t]['x']))
                y = np.concatenate((y, units[u][t]['y']))
                samples = np.concatenate((samples, units[u][t]['samples']))
        
        # sort the spike positions into bins
        x_bin = np.digitize(x, x_bins) - 1
        y_bin = np.digitize(y, y_bins) - 1

        # create the spike counts array
        spike_counts = np.zeros(positional_occupancy['occupancy'].shape)

        # sort the x and y bins into the spike_counts array
        for x_ind, y_ind in zip(x_bin, y_bin):        
            spike_counts[y_ind, x_ind] += 1

        # divide the spike counts by the occupancy
        spike_rates_by_position[u] = spike_counts / positional_occupancy['occupancy']

        # find the indices of the first position in spike_rates_by_position that is nan
        nan_ind = np.argwhere(np.isnan(spike_rates_by_position[u]))[0][0]

        # TROUBLESHOOTING - USE FIRST UNIT, CLUSTER_2
        # spike_counts[0,10] is 160, but positional_occupancy['occupancy'][0,10] is 0
        # get the x values corresponding to the 11th bin in x_bins, and the first bin in y_bins
        x_vals = np.where(x_bin == 10)[0]
        y_vals = np.where(y_bin == 0)[0]
        # get the intersection of the two
        vals = np.intersect1d(x_vals, y_vals)
        # get the spike counts for those values
        



    
    return spike_counts_by_position





if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    restricted_units = load_pickle('restricted_units', spike_dir)

    # load neuron classification data
    neuron_types_dir = os.path.join(spike_dir, 'average_waveforms') 
    neuron_types = load_pickle('neuron_types', neuron_types_dir)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # loop through units and calculate positions and various directional correlates
    # for unit in restricted_units.keys():
    #     restricted_units[unit] = get_unit_position_and_directions(dlc_data, restricted_units[unit])

    # save the restricted units
    # save_pickle(restricted_units, 'units_w_behav_correlates', spike_dir)

    # bin spikes by position
    positional_occupancy = load_pickle('positional_occupancy', dlc_dir)
    # load units
    units = load_pickle('units_w_behav_correlates', spike_dir)
    # bin spikes by position
    spike_counts_by_position = bin_spikes_by_position(units, positional_occupancy)
    