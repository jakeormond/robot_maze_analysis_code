import sys
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import platform

# if on Windows
if platform.system() == 'Windows':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# if on Linux
elif platform.system() == 'Linux':
    sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')

from utilities.get_directories import get_data_dir, get_robot_maze_directory
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import split_dictionary_by_goal
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, cm_per_pixel, get_directions_to_position, get_relative_directions_to_position
from position.calculate_occupancy import get_relative_direction_occupancy_by_position, concatenate_dlc_data, get_axes_limits, calculate_frame_durations, get_direction_bins, bin_directions
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials

import pycircstat as pycs


def rel_dir_ctrl_distribution(unit, reldir_occ_by_pos, sink_bins):
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
    rel_dir_ctrl_dist = np.zeros(len(direction_bins) -1)

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

            rel_dir_ctrl_dist = rel_dir_ctrl_dist + reldir_occ_by_pos[j,i,:] * n_spikes

    return rel_dir_ctrl_dist, n_spikes_total



def rel_dir_ctrl_distribution_all_sinks(unit, reldir_occ_by_pos, sink_bins, candidate_sinks):
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


def rel_dir_distribution(hd, positions, candidate_sink, direction_bins):
    
    directions = get_directions_to_position([candidate_sink[0], candidate_sink[1]], positions)
    relative_direction = get_relative_directions_to_position(directions, hd)
    rel_dir_binned_counts, _ = bin_directions(relative_direction, direction_bins)

    return rel_dir_binned_counts


def rel_dir_distribution_all_sinks(unit, sink_bins, candidate_sinks, direction_bins):
   
    """ 
    Create array to store the relative direcion histograms. There will be one histogram
    for each candidate consink position. The histograms will be stored in a 3D array, with
    dimensions (n_y_bins, n_x_bins, n_direction_bins). 

    PRETTY SURE THIS DOESN'T NEED THE FIRST SET OF X AND Y LOOPS!!!!!!!!!!!!!!! FIX IT!!!!!!!!!!

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
    if len(rel_dir_dist_div_ctrl.shape) > 1:
        sum_rel_dir_dist_div_ctrl = rel_dir_dist_div_ctrl.sum(axis=2)
        sum_rel_dir_dist_div_ctrl_ex = sum_rel_dir_dist_div_ctrl[:,:,np.newaxis]

    else:
        sum_rel_dir_dist_div_ctrl_ex = rel_dir_dist_div_ctrl.sum()
        
    normalised_rel_dir_dist = (rel_dir_dist_div_ctrl/sum_rel_dir_dist_div_ctrl_ex) * n_spikes_total

    return normalised_rel_dir_dist


def mean_resultant_length(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution. 
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2
    mrl = pycs.resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist)
    mean_angle = pycs.mean(dir_bin_centres, w=normalised_rel_dir_dist)

    return mrl, mean_angle



def mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins):
    """
    Calculate the mean resultant length of the normalised relative direction distribution. 
    """

    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2

    n_y_bins = normalised_rel_dir_dist.shape[0]
    n_x_bins = normalised_rel_dir_dist.shape[1]

    mrl = np.zeros((n_y_bins, n_x_bins))
    mean_angle = np.zeros((n_y_bins, n_x_bins))

    for i in range(n_y_bins):
        for j in range(n_x_bins):

            mrl[i,j] = pycs.resultant_vector_length(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])
            mean_angle[i,j] = pycs.mean(dir_bin_centres, w=normalised_rel_dir_dist[i,j,:])

    return mrl, mean_angle


def find_consink(unit, reldir_occ_by_pos, sink_bins, candidate_sinks):
    """
    Find the consink position that maximises the mean resultant length of the normalised relative direction distribution. 
    """
    #  get control occupancy distribution
    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution_all_sinks(unit, reldir_occ_by_pos, sink_bins, candidate_sinks)

    # rel dir distribution for each possible consink position
    rel_dir_dist = rel_dir_distribution_all_sinks(unit, sink_bins, candidate_sinks, direction_bins)

    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_dist, rel_dir_ctrl_dist, n_spikes_total)

    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, mean_angle = mean_resultant_length_nrdd(normalised_rel_dir_dist, direction_bins)

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angle[max_mrl_indices[0][0], max_mrl_indices[1][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle


def recalculate_consink_from_shuffle(unit, reldir_occ_by_pos_4sink, candidate_sink, direction_bins):

    rel_dir_ctrl_dist, n_spikes_total = rel_dir_ctrl_distribution(unit, reldir_occ_by_pos_4sink, sink_bins)

    # get head directions as np array
    hd = unit['hd'].to_numpy()

    x = unit['x'].to_numpy()
    y = unit['y'].to_numpy()
    positions = {'x': x, 'y': y}  

    # calculate min and max numbers of shifts
    min_shift = len(hd)//15
    max_shift = len(hd) - min_shift


    n_shuffles = 1000
    # shifts = np.random.randint(min_shift, max_shift, size=n_shuffles)
    mrl = np.zeros(n_shuffles)
    
    # args_list = [(shift, hd, positions, candidate_sink, direction_bins, rel_dir_ctrl_dist, n_spikes_total) for shift in shifts]

    # with Pool() as p:
    #     mrl = p.map(shuffle_and_calculate, args_list)
    
    for s in range(n_shuffles):

        # print every 50th shuffle
        if s % 50 == 0:
            print(f'shuffle {s}')

        # shift the hds by by a random numnber of indices between min_shift and max_shift
        shift = np.random.randint(min_shift, max_shift)
        hd_shift = np.roll(hd, shift)

        rel_dir_binned_counts = rel_dir_distribution(hd_shift, positions, candidate_sink, direction_bins)
        
        # normalise rel_dir_dist by rel_dir_ctrl_dist
        normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_binned_counts, rel_dir_ctrl_dist, n_spikes_total)

        # calculate the mean resultant length of the normalised relative direction distribution
        mrl[s], _ = mean_resultant_length(normalised_rel_dir_dist, direction_bins)

    # calculate 95% and 99.9% confidence intervals
    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    ci = (mrl_95, mrl_999)
    
    return ci


def plot_all_consinks(consinks_df, goal_coordinates, limits, jitter, plot_dir, plot_name='ConSinks'):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(plot_name)

    goals = [g for g in consinks_df.keys() if isinstance(g, int)]

    for i, g in enumerate(goals):
        ax[i].set_title(f'goal {g}')
        ax[i].set_xlabel('x position (cm)')
        ax[i].set_ylabel('y position (cm)')

        # plot the goal positions
        colours = ['b', 'g']
        for i2, g2 in enumerate(goal_coordinates.keys()):
            # draw a circle with radius 80 around the goal on ax
            circle = plt.Circle((goal_coordinates[g2][0], 
                goal_coordinates[g2][1]), 80, color=colours[i2], 
                fill=False, linewidth=5)
            ax[i].add_artist(circle)    

    # loop through the rows of the consinks_df, plot a filled red circle at the consink 
    # position if the mrl is greater than ci_999
    for cluster in consinks_df[g].index:

        x_jitter = np.random.uniform(-jitter[0], jitter[0])
        y_jitter = np.random.uniform(-jitter[1], jitter[1])

        consink_position1 = consinks_df[goals[0]].loc[cluster, 'position']
        mrl1 = consinks_df[goals[0]].loc[cluster, 'mrl']
        ci_95_1 = consinks_df[goals[0]].loc[cluster, 'ci_95']
        ci_999_1 = consinks_df[goals[0]].loc[cluster, 'ci_999']

        consink_position2 = consinks_df[goals[1]].loc[cluster, 'position']
        mrl2 = consinks_df[goals[1]].loc[cluster, 'mrl']
        ci_95_2 = consinks_df[goals[1]].loc[cluster, 'ci_95']
        ci_999_2 = consinks_df[goals[1]].loc[cluster, 'ci_999']

        if mrl1 > ci_95_1 and mrl2 > ci_95_2:           
            ax[0].plot(consink_position1[0] + x_jitter, consink_position1[1] + y_jitter, 'ro')
            ax[1].plot(consink_position2[0] + x_jitter, consink_position2[1] + y_jitter, 'ro')
            # ax[i].text(consink_position[0], consink_position[1], f'{cluster}', fontsize=12)

        elif mrl1 > ci_95_1:
             ax[0].plot(consink_position1[0] + x_jitter, consink_position1[1] + y_jitter, 'bo')

        elif mrl2 > ci_95_2:
            ax[1].plot(consink_position2[0] + x_jitter, consink_position2[1] + y_jitter, 'bo')

        for i in range(2):
            # set the x and y limits
            ax[i].set_xlim((limits['x_min']-200, limits['x_max']+200))
            ax[i].set_ylim(limits['y_min']-200, limits['y_max']+200)

            # reverse the y axis
            ax[i].invert_yaxis()

            # make the axes equal
            ax[i].set_aspect('equal')

    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    plt.show()
    pass


def shuffle_and_calculate(args):
    """
    Shuffle the head directions, calculate the relative direction distribution, and calculate the mean resultant length. 
    """
    shift, hd, positions, candidate_sink, direction_bins, rel_dir_ctrl_dist, n_spikes_total = args

    hd_shift = np.roll(hd, shift)

    rel_dir_binned_counts = rel_dir_distribution(hd_shift, positions, candidate_sink, direction_bins)
        
    # normalise rel_dir_dist by rel_dir_ctrl_dist
    normalised_rel_dir_dist = normalize_rel_dir_dist(rel_dir_binned_counts, rel_dir_ctrl_dist, n_spikes_total)

    # calculate the mean resultant length of the normalised relative direction distribution
    mrl, _ = mean_resultant_length(normalised_rel_dir_dist, direction_bins)

    return mrl
    
    

if __name__ == "__main__":
    code_to_run = [2]
    animal = 'Rat46'
    # animal = 'Rat47'
    session = '19-02-2024'
    # session = '08-02-2024'
    data_dir = get_data_dir(animal, session)
    spike_dir = os.path.join(data_dir, 'spike_sorting')

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

    ################# CALCULATE CONSINKS ###########################################
    if 0 in code_to_run:
        # load spike data
        units = load_pickle('units_by_goal', spike_dir)

        goals = units.keys()
        consinks = {}
        consinks_df = {}
        for goal in goals:
            goal_units = units[goal]
            consinks[goal] = {}
            
            for cluster in goal_units.keys():
                unit = concatenate_unit_across_trials(goal_units[cluster])
                
                # get consink  
                max_mrl, max_mrl_indices, mean_angle = find_consink(unit, reldir_occ_by_pos, sink_bins, candidate_sinks)
                consink_position = np.round([candidate_sinks['x'][max_mrl_indices[1][0]], candidate_sinks['y'][max_mrl_indices[0][0]]], 3)
                consinks[goal][cluster] = {'mrl': max_mrl, 'position': consink_position, 'mean_angle': mean_angle}

            # create a data frame with the consink positions
            consinks_df[goal] = pd.DataFrame(consinks[goal]).T

        # save consinks_df 
        save_pickle(consinks_df, 'consinks_df', spike_dir)

    
    ######################### TEST STATISTICAL SIGNIFICANCE OF CONSINKS #########################
    # shift the head directions relative to their positions, and recalculate the tuning to the 
    # previously identified consink position. 
    if 1 in code_to_run:
        # load the consinks_df
        consinks_df = load_pickle('consinks_df', spike_dir)
        # add two columns to hold the confidence intervals

        for goal in goals:
            goal_units = units[goal]
            # consinks[goal] = {}
            
            # make columns for the confidence intervals; place them directly beside the mrl column
            idx = consinks_df[goal].columns.get_loc('mrl')

            consinks_df[goal].insert(idx + 1, 'ci_95', np.nan)
            consinks_df[goal].insert(idx + 2, 'ci_999', np.nan)

            for cluster in goal_units.keys():
                unit = concatenate_unit_across_trials(goal_units[cluster])

                candidate_sink = consinks_df[goal].loc[cluster, 'position']
                # find the indices of the candidate sink in the candidate_sinks dictionaries
                sink_x_index = np.where(np.round(candidate_sinks['x'], 3) == candidate_sink[0])[0][0]
                sink_y_index = np.where(np.round(candidate_sinks['y'], 3) == candidate_sink[1])[0][0]

                reldir_occ_by_pos_4sink = reldir_occ_by_pos[:, :, sink_y_index, sink_x_index, :]

                print(f'calcualting confidence intervals for {goal} cluster {cluster}')
                ci = recalculate_consink_from_shuffle(unit, reldir_occ_by_pos_4sink, candidate_sink, direction_bins)
                consinks_df[goal].loc[cluster, 'ci_95'] = ci[0]
                consinks_df[goal].loc[cluster, 'ci_999'] = ci[1]

        save_pickle(consinks_df, 'consinks_df', spike_dir)

    ########## PLOT ALL SIGNIFICANT CONSINKS ####################################
    if 2 in code_to_run:

        # get goal coordinates
        goal_coordinates = get_goal_coordinates(data_dir=data_dir)

        # make folder consinks in spike_dir if it doesn't already exist
        plot_dir = os.path.join(spike_dir, 'consinks')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        # load the consinks_df
        consinks_df = load_pickle('consinks_df', spike_dir)

        # calculate a jitter amount to jitter the positions by so they are visible
        x_diff = np.mean(np.diff(candidate_sinks['x']))
        y_diff = np.mean(np.diff(candidate_sinks['y']))
        jitter = (x_diff/3, y_diff/3)
        plot_all_consinks(consinks_df, goal_coordinates, limits, jitter, plot_dir)
        
















    pass

