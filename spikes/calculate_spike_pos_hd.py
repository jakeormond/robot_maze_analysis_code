import os
import numpy as np
import pandas as pd
import pickle
from scipy import interpolate, ndimage

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir, get_robot_maze_directory
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import get_behaviour_dir


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

    
    # np.seterr(divide='raise', invalid='raise')

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
            # raise ValueError('There are bins with zero occupancy but non-zero spike counts.')
            pass
   
        # divide the spike counts by the occupancy
        spike_rates = np.where(positional_occupancy['occupancy'] > 0, spike_counts / positional_occupancy['occupancy'], 0.)
        # spike_rates = spike_counts / positional_occupancy['occupancy']
        indices = np.where(np.isnan(spike_rates))

        if len(indices[0]) > 0:
            print('There are nans in the spike rates for unit: ', u)
            print('The indices are: ', indices)

        spike_rates_by_position['rate_maps'][u] = np.around(spike_rates, 3)
            
    return spike_rates_by_position


def smooth_rate_maps(rate_maps):
     # make smoothed_rate_maps a copy of rate_maps
    smoothed_rate_maps = rate_maps.copy()
    occupancy = rate_maps['occupancy']

    for u in rate_maps['rate_maps'].keys():
        rate_map_copy = rate_maps['rate_maps'][u].copy()

        # make any bin with occupancy less than 1 into nan
        rate_map_copy[occupancy < 1] = np.nan

        # create a masked array
        masked_array = np.ma.array(rate_map_copy, mask=np.isnan(rate_map_copy))

        # make all the nans the average of the surrounding non-nan values.
        # this is only for smoothing, and they will all be set to white after
        # smoothing
        while True:
            for x in range(rate_map_copy.shape[1]):
                for y in range(rate_map_copy.shape[0]):
                    if np.isnan(rate_map_copy[y, x]):
                        if x == 0:
                            x_ind = [x, x+1]
                        elif x == rate_map_copy.shape[1] - 1:
                            x_ind = [x-1, x]
                        else:
                            x_ind = [x-1, x, x+1]  

                        if y == 0:
                            y_ind = [y, y+1]
                        elif y == rate_map_copy.shape[0] - 1:
                            y_ind = [y-1, y]
                        else:
                            y_ind = [y-1, y, y+1]  

                        # get the indices correspoding to rows y_ind and columns x_ind
                        x_ind, y_ind = np.meshgrid(x_ind, y_ind)

                        rate_map_copy[y, x] = np.nanmean(rate_map_copy[y_ind, x_ind])
            
            if np.isnan(rate_map_copy).sum() == 0:
                break

        rate_map_smoothed = ndimage.gaussian_filter(rate_map_copy, sigma=1)

        # use masked array to set all the nans to white
        rate_map_smoothed = np.where(masked_array.mask, np.nan, rate_map_smoothed)

        smoothed_rate_maps['rate_maps'][u] = np.around(rate_map_smoothed, 3)

    return smoothed_rate_maps


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
            # get the bin indices for each value in directional data
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



def bin_spikes_by_position_and_direction_popn(popn, directional_occupancy_by_position): 
    # get the x and y bins
    x_bins_og = directional_occupancy_by_position['x_bins']
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    y_bins_og = directional_occupancy_by_position['y_bins']
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    # get the direction bins
    direction_bins_og = directional_occupancy_by_position['direction_bins']
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    n_bins = len(direction_bins) - 1

    # loop through the units
    spike_rates_by_position_and_direction = {'popn': {}, 'x_bins': x_bins_og, 
                    'y_bins': y_bins_og, 'direction_bins': direction_bins_og}
    
    bad_vals = {}

    for p in popn.keys():
        spike_rates_by_position_and_direction['popn'][p] = \
            np.zeros((len(y_bins)-1, len(x_bins)-1, n_bins))
        

        # loop through the trials getting all the spike positions
        for i, t in enumerate(popn[p].keys()):
            # get the x and y positions
            if i == 0:
                x = popn[p][t]['x']
                y = popn[p][t]['y']
                hd = popn[p][t]['hd']
                samples = popn[p][t]['samples']
            else:
                x = np.concatenate((x, popn[p][t]['x']))
                y = np.concatenate((y, popn[p][t]['y']))
                hd = np.concatenate((hd, popn[p][t]['hd']))
                samples = np.concatenate((samples, popn[p][t]['samples']))
               
        # sort the spike positions into bins
        x_bin = np.digitize(x, x_bins) - 1
        y_bin = np.digitize(y, y_bins) - 1

        for i in range(np.max(x_bin)+1):
            for j in range(np.max(y_bin)+1):

                # get the directional occupancy for x_bin == i and y_bin == j
                directional_occupancy = directional_occupancy_by_position['occupancy'][j, i]

                # get the indices where x_bin == i and y_bin == j
                indices = np.where((x_bin == i) & (y_bin == j))[0]

                # get the head directions for these indices
                hd_temp = hd[indices]

                # sort the head directions into bins
                bin_indices = np.digitize(hd_temp, direction_bins, right=True) - 1
                bin_indices[bin_indices==-1] = 0
                # any bin_indices that are n_bins should be n_bins-1
                bin_indices[bin_indices==n_bins] = n_bins-1

                # get the spike counts and rates
                spike_counts_temp = np.zeros(n_bins)
                spike_rates_temp = np.zeros(n_bins)

                for b in range(n_bins):
                    # get the spike counts for the current bin
                    spike_counts_temp[b] = np.sum(bin_indices==b)

                    # if spike_counts_temp[b] is greater than 0 and directional_occupancy[b] is 0
                    # throw an error
                    if spike_counts_temp[b] > 0 and directional_occupancy[b] == 0:
                        # save the samples numbers and head directions to check against
                        # the dlc_data and make sure the codes not broken
                        # if bad_vals doesn't have u as a key, create it
                        if u not in bad_vals.keys():
                            bad_vals[u] = {'samples': samples[indices[bin_indices == b]], 
                                           'hd': hd[indices[bin_indices == b]]}
                        # otherwise append the samples and hd
                        else:
                            bad_vals[u]['samples'] = np.concatenate((bad_vals[u]['samples'], 
                                                                     samples[indices[bin_indices == b]]))
                            bad_vals[u]['hd'] = np.concatenate((bad_vals[u]['hd'], 
                                                                     hd[indices[bin_indices == b]]))

                    
                    if directional_occupancy[b] == 0:
                        spike_rates_temp[b] = 0.
                    else:
                        # divide the spike counts by the occupancy
                        spike_rates_temp[b] = np.round(spike_counts_temp[b] / directional_occupancy[b], 3)

                # place the spike rates in the correct position in the array
                spike_rates_by_position_and_direction['popn'][p][j, i, :] = spike_rates_temp

    return spike_rates_by_position_and_direction, bad_vals
        

def bin_spikes_by_position_and_direction_individual_units(units, directional_occupancy_by_position):
    
    # get the x and y bins
    x_bins_og = directional_occupancy_by_position['x_bins']
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    y_bins_og = directional_occupancy_by_position['y_bins']
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-1 # add a small number to the last bin so that the last value is included in the bin

    # get the direction bins
    direction_bins_og = directional_occupancy_by_position['direction_bins']
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    n_bins = len(direction_bins) - 1

    # loop through the units
    spike_rates_by_position_and_direction = {'units': {}, 'x_bins': x_bins_og, 
                    'y_bins': y_bins_og, 'direction_bins': direction_bins_og}
    
    bad_vals = {}
    
    for u in units.keys():

        spike_rates_by_position_and_direction['units'][u] = \
            np.zeros((len(y_bins)-1, len(x_bins)-1, n_bins))
        
        # loop through the trials getting all the spike positions
        for i, t in enumerate(units[u].keys()):
            # get the x and y positions
            if i == 0:
                x = units[u][t]['x']
                y = units[u][t]['y']
                hd = units[u][t]['hd']
                samples = units[u][t]['samples']
            else:
                x = np.concatenate((x, units[u][t]['x']))
                y = np.concatenate((y, units[u][t]['y']))
                hd = np.concatenate((hd, units[u][t]['hd']))
                samples = np.concatenate((samples, units[u][t]['samples']))
               
        # sort the spike positions into bins
        x_bin = np.digitize(x, x_bins) - 1
        y_bin = np.digitize(y, y_bins) - 1

        for i in range(np.max(x_bin)+1):
            for j in range(np.max(y_bin)+1):

                # get the directional occupancy for x_bin == i and y_bin == j
                directional_occupancy = directional_occupancy_by_position['occupancy'][j, i]

                # get the indices where x_bin == i and y_bin == j
                indices = np.where((x_bin == i) & (y_bin == j))[0]

                # get the head directions for these indices
                hd_temp = hd[indices]

                # sort the head directions into bins
                bin_indices = np.digitize(hd_temp, direction_bins, right=True) - 1
                bin_indices[bin_indices==-1] = 0
                # any bin_indices that are n_bins should be n_bins-1
                bin_indices[bin_indices==n_bins] = n_bins-1

                # get the spike counts and rates
                spike_counts_temp = np.zeros(n_bins)
                spike_rates_temp = np.zeros(n_bins)

                for b in range(n_bins):
                    # get the spike counts for the current bin
                    spike_counts_temp[b] = np.sum(bin_indices==b)

                    # if spike_counts_temp[b] is greater than 0 and directional_occupancy[b] is 0
                    # throw an error
                    if spike_counts_temp[b] > 0 and directional_occupancy[b] == 0:
                        # save the samples numbers and head directions to check against
                        # the dlc_data and make sure the codes not broken
                        # if bad_vals doesn't have u as a key, create it
                        if u not in bad_vals.keys():
                            bad_vals[u] = {'samples': samples[indices[bin_indices == b]], 
                                           'hd': hd[indices[bin_indices == b]]}
                        # otherwise append the samples and hd
                        else:
                            bad_vals[u]['samples'] = np.concatenate((bad_vals[u]['samples'], 
                                                                     samples[indices[bin_indices == b]]))
                            bad_vals[u]['hd'] = np.concatenate((bad_vals[u]['hd'], 
                                                                     hd[indices[bin_indices == b]]))

                    
                    if directional_occupancy[b] == 0:
                        spike_rates_temp[b] = 0.
                    else:
                        # divide the spike counts by the occupancy
                        spike_rates_temp[b] = np.round(spike_counts_temp[b] / directional_occupancy[b], 3)

                # place the spike rates in the correct position in the array
                spike_rates_by_position_and_direction['units'][u][j, i, :] = spike_rates_temp

    return spike_rates_by_position_and_direction, bad_vals


def check_bad_vals(bad_vals, dlc_data):

    # concatenate the dlc_data
    for i, d in enumerate(dlc_data.keys()):
        if i == 0:
            dlc_data_concat = dlc_data[d]
        else:
            dlc_data_concat = pd.concat([dlc_data_concat, dlc_data[d]], ignore_index=True)
   
    # loop through the units
    for u in bad_vals.keys():
        # loop through the samples
        for i, s in enumerate(bad_vals[u]['samples']):
            # get the index of the sample in dlc_data_concat that is smaller but closest to s
            ind1 = dlc_data_concat['video_samples'][dlc_data_concat['video_samples'] < s].idxmax()
            # get the index of the sample in dlc_data_concat that is larger but closest to s
            ind2 = dlc_data_concat['video_samples'][dlc_data_concat['video_samples'] > s].idxmin()

            # get the head direction from dlc_data_concat at ind1 and ind2
            hd1 = dlc_data_concat['hd'][ind1]
            hd2 = dlc_data_concat['hd'][ind2]

            # get the hd at i from bad values
            hd = bad_vals[u]['hd'][i]

            # print the values for manual inspection
            print('hd before: ', hd1)
            print('hd after: ', hd2)
            print('hd from bad_vals: ', hd)
    pass


def create_artificial_unit(units, directional_occupancy_by_position):

    pass

    return 


if __name__ == "__main__":
    animal = 'Rat47'
    session = '16-02-2024'
    data_dir = get_data_dir(animal, session)

    # code_to_run = range(8)
    code_to_run = [4]


    ################## LOAD DATA ##################
    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    restricted_units = load_pickle('restricted_units', spike_dir)

    # load neuron classification data
    # neuron_types = load_pickle('neuron_types', spike_dir)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # load occupancy data
    positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

    # find any NaNs or infs in the positional occupancy['occupancy']
    if np.isnan(positional_occupancy['occupancy']).sum() > 0:
        raise ValueError('There are NaNs in the positional occupancy.')
    if np.isinf(positional_occupancy['occupancy']).sum() > 0:
        raise ValueError('There are infs in the positional occupancy.')


    ####################### CALCULATE SPIKE POSITIONS AND DIRECTIONS #######################
    # loop through units and calculate positions and various directional correlates
    if 0 in code_to_run:
        for unit in restricted_units.keys():
            restricted_units[unit] = get_unit_position_and_directions(dlc_data, restricted_units[unit])

        # # save the restricted units
        save_pickle(restricted_units, 'units_w_behav_correlates', spike_dir)
        del restricted_units


    ################## LOAD UNITS (IF NOT RUNNING CODE ABOVE) ##################
    # load units
    units = load_pickle('units_w_behav_correlates', spike_dir)


    ############################### SINGLE RATE MAPS FOR ENTIRE SESSION ###############################
    if 1 in code_to_run:
        # bin spikes by position
        rate_maps = bin_spikes_by_position(units, positional_occupancy)
        # save the spike counts by position
        save_pickle(rate_maps, 'rate_maps', spike_dir)

        # create smoothed rate_maps
        smoothed_rate_maps = smooth_rate_maps(rate_maps)
        save_pickle(smoothed_rate_maps, 'smoothed_rate_maps', spike_dir) 


    ############################### DIRECTIONAL TUNING FOR ENTIRE SESSION ###############################
    if 2 in code_to_run:
        # bin spikes by direction
        directional_occupancy = load_pickle('directional_occupancy', dlc_dir)
        spike_rates_by_direction, spike_counts = bin_spikes_by_direction(units, 
                                                directional_occupancy)
        # save the spike counts and rates by direction
        save_pickle(spike_rates_by_direction, 'spike_rates_by_direction', spike_dir)
        save_pickle(spike_counts, 'spike_counts_by_direction', spike_dir)


    ####################SPIKE RATES BY POSITION AND DIRECTION FOR ENTIRE SESSSION ###############################
    if 3 in code_to_run:
        # load the directional occupancy by position data
        directional_occupancy_by_position = load_pickle('directional_occupancy_by_position', dlc_dir)
        # bin spikes by position and direction
        spike_rates_by_position_and_direction, bad_vals = bin_spikes_by_position_and_direction_individual_units(units, 
                                                directional_occupancy_by_position)
        
        check_bad_vals(bad_vals, dlc_data)

        # save the spike rates by position and direction
        save_pickle(spike_rates_by_position_and_direction, 'spike_rates_by_position_and_direction', spike_dir)


    ################################### ARTIFICIAL UNIT - NOT DONE YET ###################################
    # create an artificial unit for testing vector field code
    # artifial_unit = create_artificial_unit(units, directional_occupancy_by_position)



    ############################ DATA SEPARATED BY GOAL ############################
    ##################################################################################
    if 4 in code_to_run:
        # sort spike data by goal
        behaviour_dir = get_behaviour_dir(data_dir)
        behaviour_data_by_goal = load_pickle('behaviour_data_by_goal', behaviour_dir)

        units_by_goal = sort_units_by_goal(behaviour_data_by_goal, units)

        # save the units by goal
        save_pickle(units_by_goal, 'units_by_goal', spike_dir)
        del units_by_goal


    ############################# LOAD UNITS BY GOAL ################################
    units_by_goal = load_pickle('units_by_goal', spike_dir)
  

    ######################### RATE MAPS BY GOAL ############################
    if 5 in code_to_run:
        # load the positional occupancy data by goal
        positional_occupancy_by_goal = load_pickle('positional_occupancy_by_goal', dlc_dir)

        # bin spikes by position by goal
        rate_maps_by_goal = {}
        smoothed_rate_maps_by_goal = {}
        for g in units_by_goal.keys():
            rate_maps_by_goal[g] = bin_spikes_by_position(units_by_goal[g], positional_occupancy_by_goal[g])
            smoothed_rate_maps_by_goal[g] = smooth_rate_maps(rate_maps_by_goal[g])

        save_pickle(rate_maps_by_goal, 'rate_maps_by_goal', spike_dir)  
        del rate_maps_by_goal
        save_pickle(smoothed_rate_maps_by_goal, 'smoothed_rate_maps_by_goal', spike_dir)
        del smoothed_rate_maps_by_goal


    ######################### SPIKE RATES BY DIRECTION BY GOAL ############################
    if 6 in code_to_run:
        # load the directional occupancy data by goal
        directional_occupancy_by_goal = load_pickle('directional_occupancy_by_goal', dlc_dir)

        # bin spikes by direction by goal
        spike_rates_by_direction_by_goal = {}
        spike_counts_by_direction_by_goal = {}
        for g in units_by_goal.keys():
            spike_rates_by_direction_by_goal[g], spike_counts_by_direction_by_goal[g]\
                = bin_spikes_by_direction(units_by_goal[g], directional_occupancy_by_goal[g])

        save_pickle(spike_rates_by_direction_by_goal, 'spike_rates_by_direction_by_goal', spike_dir)
        del spike_rates_by_direction_by_goal


    ######################### SPIKE RATES BY POSITION AND DIRECTION BY GOAL ############################
    if 7 in code_to_run:
        # load the directional occupancy by position data by goal
        directional_occupancy_by_position_by_goal = load_pickle('directional_occupancy_by_position_by_goal', dlc_dir)

        spike_rates_by_position_and_direction_by_goal = {}
        for i, g in enumerate(units_by_goal.keys()):  
            spike_rates_by_position_and_direction_by_goal[g], bad_vals = \
                bin_spikes_by_position_and_direction_individual_units(units_by_goal[g], 
                directional_occupancy_by_position_by_goal[g])

            if i == 0:
                spike_rates_by_position_and_direction_by_goal['x_bins'] = \
                    spike_rates_by_position_and_direction_by_goal[g]['x_bins']
                spike_rates_by_position_and_direction_by_goal['y_bins'] = \
                    spike_rates_by_position_and_direction_by_goal[g]['y_bins']
                spike_rates_by_position_and_direction_by_goal['direction_bins'] = \
                    spike_rates_by_position_and_direction_by_goal[g]['direction_bins']  

            # remove the x_bins, y_bins and direction_bins from the spike_rates_by_position_and_direction_by_goal[g]
            spike_rates_by_position_and_direction_by_goal[g].pop('x_bins')
            spike_rates_by_position_and_direction_by_goal[g].pop('y_bins')
            spike_rates_by_position_and_direction_by_goal[g].pop('direction_bins')       

            spike_rates_by_position_and_direction_by_goal[g] = spike_rates_by_position_and_direction_by_goal[g]['units']
          
        save_pickle(spike_rates_by_position_and_direction_by_goal, 'spike_rates_by_position_and_direction_by_goal', spike_dir)
        del spike_rates_by_position_and_direction_by_goal


    ######################################## COMBINE UNITS BY GOAL ############################################
    if 8 in code_to_run:
        popn_by_goal = {}
        for g in units_by_goal.keys():
            units = list(units_by_goal[g].keys())
            trials = list(units_by_goal[g][units[0]].keys())
            
            popn_by_goal[g] = {}
            popn_by_goal[g]['pyramidal'] = {}
            popn_by_goal[g]['interneuron'] = {}

            for t in trials:
                for u in units:
                    if neuron_types[u] == 'pyramidal':
                        if t not in popn_by_goal[g]['pyramidal'].keys():
                            popn_by_goal[g]['pyramidal'][t] = units_by_goal[g][u][t]
                        else:
                            popn_by_goal[g]['pyramidal'][t] = pd.concat([popn_by_goal[g]['pyramidal'][t], 
                                units_by_goal[g][u][t]], ignore_index=True)
                    else:
                        if t not in popn_by_goal[g]['interneuron'].keys():
                            popn_by_goal[g]['interneuron'][t] = units_by_goal[g][u][t]
                        else:
                            popn_by_goal[g]['interneuron'][t] = pd.concat([popn_by_goal[g]['interneuron'][t], 
                                units_by_goal[g][u][t]], ignore_index=True)

        save_pickle(popn_by_goal, 'popn_by_goal', spike_dir)
        del popn_by_goal



    ################### SPIKE RATES BY POSITION AND DIRECTION BY GOAL - COMBINED PRINCIPAL CELLS ############################
    if 9 in code_to_run:
        # load the popn data
        popn_by_goal = load_pickle('popn_by_goal', spike_dir)

        # load the directional occupancy by position data by goal
        directional_occupancy_by_position_by_goal = load_pickle('directional_occupancy_by_position_by_goal', dlc_dir)

        spike_rates_by_position_and_direction_by_goal_popn = {}
        for g in units_by_goal.keys():  
            spike_rates_by_position_and_direction_by_goal_popn[g], bad_vals = \
                bin_spikes_by_position_and_direction_popn(popn_by_goal[g], 
                directional_occupancy_by_position_by_goal[g])
            
        save_pickle(spike_rates_by_position_and_direction_by_goal_popn, 'spike_rates_by_position_and_direction_by_goal_popn', spike_dir)



    ################### SPIKE RATES BY POSITION AND DIRECTION BY GOAL - COMBINED PRINCIPAL CELLS V2 ############################
    if 10 in code_to_run:
        # load the individual SPIKE RATES BY POSITION AND DIRECTION BY GOAL 
        spike_rates_by_position_and_direction_by_goal = load_pickle('spike_rates_by_position_and_direction_by_goal', spike_dir)

        goals = [g for g in spike_rates_by_position_and_direction_by_goal.keys() if isinstance(g, int)]

        spike_rates_by_position_and_direction_by_goal_popn = {}

        spike_rates_by_position_and_direction_by_goal_popn['x_bins'] = \
            spike_rates_by_position_and_direction_by_goal['x_bins']
        spike_rates_by_position_and_direction_by_goal_popn['y_bins'] = \
            spike_rates_by_position_and_direction_by_goal['y_bins']
        spike_rates_by_position_and_direction_by_goal_popn['direction_bins'] = \
            spike_rates_by_position_and_direction_by_goal['direction_bins']

        for g in goals:
            spike_rates_by_position_and_direction_by_goal_popn[g] = {}
            spike_rates_by_position_and_direction_by_goal_popn[g]= {}

        # units = list(neuron_types.keys())
        # units_temp = [8, 16, 18, 19, 29, 53, 59, 69, 75, 109, 112, 113, 186, 189, 
        #          193, 204, 205, 211, 229, 232, 234, 254, 274, 287, 290, 291]
        
        # units_temp = [8, 16, 18, 19, 29, 53, 59, 69, 75, 109, 112, 113, 186, 189, 
        #          193, 204, 205, 211, 229, 232, 234, 254, 274, 287, 290, 291]
        
        # units_temp = [109, 113]
        
        # units = [f'cluster_{u}' for u in units_temp]    
        


        for u in units:
            for g in goals:
                if neuron_types[u] not in spike_rates_by_position_and_direction_by_goal_popn[g].keys():
                    spike_rates_by_position_and_direction_by_goal_popn[g][neuron_types[u]] = \
                        spike_rates_by_position_and_direction_by_goal[g]['units'][u]
                else:
                    # add the spike rates to the existing spike rates
                    spike_rates_by_position_and_direction_by_goal_popn[g][neuron_types[u]] += \
                        spike_rates_by_position_and_direction_by_goal[g]['units'][u]
                        
        save_pickle(spike_rates_by_position_and_direction_by_goal_popn, 'spike_rates_by_position_and_direction_by_goal_popn', spike_dir)
                        
         
         



    pass
    