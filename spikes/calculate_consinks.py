import sys
import os
import numpy as np

import platform

# if on Windows
if platform.system() == 'Windows':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# if on Linux
elif platform.system() == 'Linux':
    sys.path.append('/home/Jake/Documents/python_code/robot_maze_analysis_code')

from utilities.get_directories import get_data_dir, get_robot_maze_directory
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import split_dictionary_by_goal
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, cm_per_pixel, get_directions_to_position, get_relative_directions_to_position
from position.calculate_occupancy import get_relative_direction_occupancy_by_position, concatenate_dlc_data, get_axes_limits, calculate_frame_durations, get_direction_bins, bin_directions
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials

import pycircstat as pycs


def rel_dir_ctrl_distribution(unit, reldir_occ_by_pos, sink_bins, candidate_sinks):
    """
    For a given unit, produces relative direction occupancy distributions 
    for each candidate consink position based on the number of spikes fired 
    at each positional bin. 
    """


    # get head directions as np array
    hd = unit['hd'].to_numpy()

    x = unit['x'].to_numpy()
    y = unit['y'].to_numpy()

    x_bin = np.digitize(x, sink_bins['x']) - 1
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    x_bin[x_bin == (len(sink_bins['x'])-1)] = len(sink_bins['x'])-2

    y_bin = np.digitize(y, sink_bins['y']) - 1
    # find y_bin == n_y_bins, and set it to n_y_bins - 1
    y_bin[y_bin == (len(sink_bins['y'])-1)] = len(sink_bins['y'])-2

    direction_bins = get_direction_bins(n_bins=12)
    rel_dir_ctrl_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) -1))

    # loop through the x and y bins
    n_spikes_total = 0
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            n_spikes = len(indices)
            if n_spikes == 0:
                continue
            n_spikes_total = n_spikes_total + n_spikes

            rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[j,i,:,:,:] * n_spikes

    return rel_dir_ctrl_dist, n_spikes_total


def rel_dir_distribution(unit, sink_bins, candidate_sinks, direction_bins):
   
    """ 
    Create array to store the relative direcion histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_y_bins, n_x_bins, n_direction_bins). 
    """

    # create array to store relative direction histograms
    rel_dir_dist = np.zeros((len(candidate_sinks['y']), len(candidate_sinks['x']), len(direction_bins) - 1))  
   
    # get head directions as np array
    hd = unit['hd'].to_numpy()

    x = unit['x'].to_numpy()
    y = unit['y'].to_numpy()

    x_bin = np.digitize(x, sink_bins['x']) - 1
    # find x_bin == n_x_bins, and set it to n_x_bins - 1
    x_bin[x_bin == (len(sink_bins['x'])-1)] = len(sink_bins['x'])-2

    y_bin = np.digitize(y, sink_bins['y']) - 1
    # find y_bin == n_y_bins, and set it to n_y_bins - 1
    y_bin[y_bin == (len(sink_bins['y'])-1)] = len(sink_bins['y'])-2

    # loop through the x and y bins
    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            x_positions = x[indices]
            y_positions = y[indices]
            positions = {'x': x_positions, 'y': y_positions}

            # get the head directions for these indices
            hd_temp = hd[indices]

            # loop through candidate consink positions
            for i2, x_sink in enumerate(candidate_sinks['x']):
                for j2, y_sink in enumerate(candidate_sinks['y']):

                    # get directions to sink                    
                    directions = get_directions_to_position([x_sink, y_sink], positions)

                    # get the relative direction
                    relative_direction = get_relative_directions_to_position(directions, hd_temp)

                    # bin the relative directions 
                    rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)
                    rel_dir_dist[j2, i2, :] = rel_dir_dist[j2, i2, :] + rel_dir_binned_counts
    
    return rel_dir_dist


def normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total):
    """
    Normalise the relative direction distribution by the control distribution. 
    """

    # first, divide rel_dir_dist by rel_dir_ctrl_dist
    rel_dir_dist_div_ctrl = rel_dir_dist/rel_dir_ctrl_dist 

    # now we want the counts in each histogram to sum to the total number of spikes
    sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=2)
    sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:,:,np.newaxis]
    normalised_rel_dir_dist = (rel_dir_dist_div_ctrl/sum_rel_dir_dist_div_ctrl_ex) * n_spikes_total

    return normalised_rel_dir_dist


def mean_resultant_length(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution. 
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2

    n_y_bins = normalised_rel_dir_dist.shape[0]
    n_x_bins = normalised_rel_dir_dist.shape[1]

    mrl = np.zeros((n_y_bins, n_x_bins))

    for i in range(n_y_bins):
        for j in range(n_x_bins):

            mrl[i,j] = pycs.resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])

    return mrl


if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    # get direction bins
    direction_bins = get_direction_bins(n_bins=12)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)
    dlc_data = calculate_frame_durations(dlc_data)
    dlc_data_concat = concatenate_dlc_data(dlc_data)

    # get x and y limits
    limits = get_axes_limits(dlc_data_concat)

    # get relative direction occupancy by position if np array not already saved
    if os.path.exists(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy')) == False:
        reldir_occ_by_pos, sink_bins, candidate_sinks = get_relative_direction_occupancy_by_position(dlc_data_concat, limits)
        np.save(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy'), reldir_occ_by_pos)
        # save sink bins and candidate sinks as pickle files
        save_pickle(sink_bins, 'sink_bins', dlc_dir)
        save_pickle(candidate_sinks, 'candidate_sinks', dlc_dir)     

    else:
        reldir_occ_by_pos = np.load(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy'))
        sink_bins = load_pickle('sink_bins', dlc_dir)
        candidate_sinks = load_pickle('candidate_sinks', dlc_dir)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_by_goal', spike_dir)

    goals = units.keys()
    for goal in goals:
        goal_units = units[goal]
        
        for cluster in goal_units.keys():
            unit = concatenate_unit_across_trials(goal_units[cluster])

            #  get control occupancy distribution
            rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution(unit, reldir_occ_by_pos, sink_bins, candidate_sinks)

            # rel dir distribution for each possible consink position
            rel_dir_dist = rel_dir_distribution(unit, sink_bins, candidate_sinks, direction_bins)

            # normalise rel_dir_dist by rel_dir_ctrl_dist
            normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)

            # calculate the mean resultant length of the normalised relative direction distribution
            mrl = mean_resultant_length(normalised_rel_dir_dist, direction_bins)

            mean(alpha, w=None, ci=None, d=None, axis=None, axial_correction=1)





    pass