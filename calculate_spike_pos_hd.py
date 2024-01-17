import os
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate

from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from load_behaviour import get_behaviour_dir

def sort_units_by_goal(behaviour_data_by_goal, units):

    units_by_goal = {}

    for g in behaviour_data_by_goal.keys():
        units_by_goal[g] = {}
        
        for u in units.keys():
            units_by_goal[g][u] = {}
            
            for t in behaviour_data_by_goal[g].keys():
                units_by_goal[g][u][t] = units[u][t]

    return units_by_goal

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
    x_bins_og = positional_occupancy['x_bins']
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins_og[-1] + 1

    y_bins_og = positional_occupancy['y_bins']
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins_og[-1] + 1

    # loop through the units
    spike_rates_by_position = {}
    spike_rates_by_position['x_bins'] = x_bins_og
    spike_rates_by_position['y_bins'] = y_bins_og
    spike_rates_by_position['occupancy'] = positional_occupancy['occupancy']
    spike_rates_by_position['rate_maps'] = {}

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

        # get the indices of any bins that have non-zero spike counts but zero occupancy
        # these are the bins that have no occupancy, but have spikes
        zero_occupancy_ind = np.argwhere((spike_counts > 0) & 
                                         (positional_occupancy['occupancy'] == 0))

        # throw an error if there are any
        if zero_occupancy_ind.size > 0:
            raise ValueError('There are bins with zero occupancy but non-zero spike counts.')
        
        # divide the spike counts by the occupancy
        spike_rates = spike_counts / positional_occupancy['occupancy']
        spike_rates_by_position['rate_maps'][u] = np.around(spike_rates, 3)
            
    return spike_rates_by_position


def bin_spikes_by_direction(units, directional_occupancy):
    
    direction_bins_og = directional_occupancy['bins']
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    n_bins = len(direction_bins) - 1

    occupancy = directional_occupancy['occupancy']

    # directional occupancy is split into 2 groups, allocentric
    # and egocentric. For now, we will only look at 'hd' (i.e. head
    # direction) from the allocentric group. We will use all
    # the egocentric groups, which consists of head direction
    # relative to the goals and the tv screens. 
    occupancy['hd'] = occupancy['allocentric']['hd']
    for d in occupancy['egocentric'].keys():
        occupancy[d] = occupancy['egocentric'][d]
    # remove the egocentric and allocentric keys
    occupancy.pop('egocentric')
    occupancy.pop('allocentric')

    # create dictionaries for the spike counts and rates
    spike_counts_temp = {}
    spike_rates_temp = {}

    for u in units.keys():

        spike_counts_temp[u] = {}
        spike_rates_temp[u] = {}

        # loop through the different types of directional data
        for d in occupancy.keys():       
            # loop through the trials 
            for i, t in enumerate(units[u].keys()):
                
                if i == 0:
                    directional_data = units[u][t][d]
                    
                else:
                    directional_data = np.concatenate((directional_data, units[u][t][d]))
            
            # sort the directional data into bins
            # get the bin indices for each value in dlc_data[d]
            bin_indices = np.digitize(directional_data, direction_bins, right=True) - 1
            # any bin_indices that are -1 should be 0
            bin_indices[bin_indices==-1] = 0
            # any bin_indices that are n_bins should be n_bins-1
            bin_indices[bin_indices==n_bins] = n_bins-1

            # get the spike counts and rates
            spike_counts_temp[u][d] = np.zeros(n_bins)
            spike_rates_temp[u][d] = np.zeros(n_bins)

            for i in range(n_bins):
                # get the spike counts for the current bin
                spike_counts_temp[u][d][i] = np.sum(bin_indices==i)
                # divide the spike counts by the occupancy
                spike_rates_temp[u][d][i] = np.round(spike_counts_temp[u][d][i] / occupancy[d][i], 3)

    # create the spike counts and rates dictionaries
    spike_counts = {}
    spike_rates = {}
    
    spike_counts['units'] = spike_counts_temp
    spike_rates['units'] = spike_rates_temp

    spike_counts['bins'] = direction_bins_og
    spike_rates['bins'] = direction_bins_og

    spike_counts['occupancy'] = occupancy
    spike_rates['occupancy'] = occupancy
   
    return spike_rates, spike_counts


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
    rate_maps = bin_spikes_by_position(units, positional_occupancy)
    # save the spike counts by position
    save_pickle(rate_maps, 'rate_maps', spike_dir)

    # bin spikes by direction
    directional_occupancy = load_pickle('directional_occupancy', dlc_dir)
    spike_rates_by_direction, spike_counts = bin_spikes_by_direction(units, 
                                            directional_occupancy)
    # save the spike counts and rates by direction
    save_pickle(spike_rates_by_direction, 'spike_rates_by_direction', spike_dir)
    save_pickle(spike_counts, 'spike_counts_by_direction', spike_dir)

    # sort spike data by goal
    behaviour_dir = get_behaviour_dir(data_dir)
    behaviour_data_by_goal = load_pickle('behaviour_data_by_goal', behaviour_dir)

    units_by_goal = sort_units_by_goal(behaviour_data_by_goal, units)

    # load the positional occupancy data by goal
    positional_occupancy_by_goal = load_pickle('positional_occupancy_by_goal', dlc_dir)

    # bin spikes by position by goal
    rate_maps_by_goal = {}
    for g in units_by_goal.keys():
        rate_maps_by_goal[g] = bin_spikes_by_position(units_by_goal[g], positional_occupancy_by_goal[g])
    
    save_pickle(rate_maps_by_goal, 'rate_maps_by_goal', spike_dir)  

    # load the directional occupancy data by goal
    directional_occupancy_by_goal = load_pickle('directional_occupancy_by_goal', dlc_dir)

    # bin spikes by direction by goal
    spike_rates_by_direction_by_goal = {}
    spike_counts_by_direction_by_goal = {}
    for g in units_by_goal.keys():
        spike_rates_by_direction_by_goal[g], spike_counts_by_direction_by_goal[g]\
              = bin_spikes_by_direction(units_by_goal[g], directional_occupancy_by_goal[g])

    save_pickle(spike_rates_by_direction_by_goal, 'spike_rates_by_direction_by_goal', spike_dir)

    pass
    