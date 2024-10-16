import sys
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

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
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, get_directions_to_position, get_relative_directions_to_position
from position.calculate_occupancy import get_relative_direction_occupancy_by_position, concatenate_dlc_data, get_axes_limits, calculate_frame_durations, get_direction_bins, bin_directions
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials

import pycircstat as pycs
from scipy.stats import wilcoxon

cm_per_pixel = 0.2


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


def find_consink(unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks):
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

    # find any nans in mrl and set them to 0
    mrl[np.isnan(mrl)] = 0

    # find the maximum mrl, and its indices
    max_mrl = np.max(mrl)
    max_mrl_indices = np.where(mrl == max_mrl)
    mean_angle = np.round(mean_angle[max_mrl_indices[0][0], max_mrl_indices[1][0]], 3)

    return np.round(max_mrl, 3), max_mrl_indices, mean_angle


def calculate_shift_mrl(hd, min_shift, max_shift, unit, reldir_occ_by_pos, sink_bins, candidate_sinks):
    shift = np.random.randint(min_shift, max_shift)
    hd_shift = np.roll(hd, shift)
    shifted_unit = unit.copy()
    shifted_unit['hd'] = hd_shift

    mrl, _, _ = find_consink(shifted_unit, reldir_occ_by_pos, sink_bins, candidate_sinks)
    return mrl


def recalculate_consink_to_all_candidates_from_shuffle(unit, reldir_occ_by_pos, sink_bins, candidate_sinks):

    hd = unit['hd'].to_numpy()

    # calculate min and max numbers of shifts
    min_shift = len(hd)//15
    max_shift = len(hd) - min_shift

    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)

    mrl = Parallel(n_jobs=-1, verbose=50)(delayed(calculate_shift_mrl)(hd, min_shift, max_shift, unit, reldir_occ_by_pos, sink_bins, candidate_sinks) for s in range(n_shuffles))

    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    ci = (mrl_95, mrl_999)
    
    return ci


def recalculate_consink_to_single_candidate_from_shuffle(unit, reldir_occ_by_pos_4sink, candidate_sink, direction_bins):

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


def plot_all_consinks_2goals(consinks_df, goal_coordinates, limits, jitter, plot_dir, plot_name='ConSinks'):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(plot_name, fontsize=24)

    goals = [g for g in consinks_df.keys() if isinstance(g, int)]
    colours = ['g', 'g']


    for i, g in enumerate(goals):
        ax[i].set_title(f'goal {g}', fontsize=20)
        ax[i].set_xlabel('x position (cm)', fontsize=16)
        ax[i].set_ylabel('y position (cm)', fontsize=16)

        # plot the goal positions
        circle = plt.Circle((goal_coordinates[g][0], 
                goal_coordinates[g][1]), 80, color=colours[i], 
                fill=False, linewidth=5)
        ax[i].add_artist(circle)   
        
        # for i2, g2 in enumerate(goal_coordinates.keys()):
        #     # draw a circle with radius 80 around the goal on ax
        #     circle = plt.Circle((goal_coordinates[g2][0], 
        #         goal_coordinates[g2][1]), 80, color=colours[i2], 
        #         fill=False, linewidth=5)
        #     ax[i].add_artist(circle)    

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
            # ax[0].plot(consink_position1[0] + x_jitter, consink_position1[1] + y_jitter, 'ro')
            # ax[1].plot(consink_position2[0] + x_jitter, consink_position2[1] + y_jitter, 'ro')
            # ax[i].text(consink_position[0], consink_position[1], f'{cluster}', fontsize=12)

            circle = plt.Circle((consink_position1[0] + x_jitter, 
                consink_position1[1] + y_jitter), 60, color='r', 
                fill=True)
            ax[0].add_artist(circle)  
            circle = plt.Circle((consink_position2[0] + x_jitter, 
                consink_position2[1] + y_jitter), 60, color='r', 
                fill=True)
            ax[1].add_artist(circle)   

        elif mrl1 > ci_95_1:
            # ax[0].plot(consink_position1[0] + x_jitter, consink_position1[1] + y_jitter, 'bo')
            circle = plt.Circle((consink_position1[0] + x_jitter, 
                consink_position1[1] + y_jitter), 60, color='b', 
                fill=True)
            ax[0].add_artist(circle)  

        elif mrl2 > ci_95_2:
            # ax[1].plot(consink_position2[0] + x_jitter, consink_position2[1] + y_jitter, 'bo')
            circle = plt.Circle((consink_position2[0] + x_jitter, 
                consink_position2[1] + y_jitter), 60, color='b', 
                fill=True)
            ax[1].add_artist(circle)

        for i in range(2):
            # set the x and y limits
            ax[i].set_xlim((limits['x_min']-200, limits['x_max']+200))
            ax[i].set_ylim(limits['y_min']-200, limits['y_max']+200)

            # reverse the y axis
            ax[i].invert_yaxis()

            # make the axes equal
            ax[i].set_aspect('equal')

            # set font size of axes
            ax[i].tick_params(axis='both', which='major', labelsize=14)

            # get the axes values
            x_ticks = ax[i].get_xticks()
            y_ticks = ax[i].get_yticks()

            # convert the axes values to cm
            x_ticks_cm = x_ticks * cm_per_pixel
            y_ticks_cm = y_ticks * cm_per_pixel

            # set the axes values to cm
            ax[i].set_xticklabels(x_ticks_cm)
            ax[i].set_yticklabels(y_ticks_cm)


    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    plt.show()
    pass


def plot_all_consinks(consinks_df, goal_coordinates, limits, jitter, plot_dir, plot_name='ConSinks'):
    
    # create a fig with 1 plot
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle(plot_name, fontsize=24)

    ax.set_xlabel('x position (cm)', fontsize=16)
    ax.set_ylabel('y position (cm)', fontsize=16)

    # plot the goal positions
    circle = plt.Circle((goal_coordinates[0], 
            goal_coordinates[1]), 80, color='g', 
            fill=False, linewidth=5)
    ax.add_artist(circle)   
        
    # loop through the rows of the consinks_df, plot a filled red circle at the consink 
    # position if the mrl is greater than ci_999

    clusters = []
    consink_positions = []

    for cluster in consinks_df.index:

        

        x_jitter = np.random.uniform(-jitter[0], jitter[0])
        y_jitter = np.random.uniform(-jitter[1], jitter[1])

        consink_position = consinks_df.loc[cluster, 'position']
        mrl = consinks_df.loc[cluster, 'mrl']
        ci_95 = consinks_df.loc[cluster, 'ci_95']
        ci_999 = consinks_df.loc[cluster, 'ci_999']

        if mrl > ci_95:           
            
            circle = plt.Circle((consink_position[0] + x_jitter, 
                consink_position[1] + y_jitter), 60, color='r', 
                fill=True)
            ax.add_artist(circle)  

            clusters.append(cluster)
            consink_positions.append(consink_position)
          
        for i in range(2):
            # set the x and y limits
            ax.set_xlim((limits['x_min']-200, limits['x_max']+200))
            ax.set_ylim(limits['y_min']-200, limits['y_max']+200)

            # reverse the y axis
            ax.invert_yaxis()

            # make the axes equal
            ax.set_aspect('equal')

            # set font size of axes
            ax.tick_params(axis='both', which='major', labelsize=14)

            # get the axes values
            x_ticks = ax.get_xticks()
            y_ticks = ax.get_yticks()

            # convert the axes values to cm
            x_ticks_cm = x_ticks * cm_per_pixel
            y_ticks_cm = y_ticks * cm_per_pixel

            # set the axes values to cm
            ax.set_xticklabels(x_ticks_cm)
            ax.set_yticklabels(y_ticks_cm)


    plt.savefig(os.path.join(plot_dir, plot_name + '.png'))
    plt.show()

    # make df with clusters and cluster_positions
    consinks = pd.DataFrame({'cluster': clusters, 'position': consink_positions})
    pass


def restrict_to_significant_consinks(consinks_df):
    for g in consinks_df.keys():
        consinks_df[g] = consinks_df[g][(consinks_df[g]['mrl'] > consinks_df[g]['ci_95'])]  

    return consinks_df



def calculate_consink_distance_to_goal(consinks_df, goal_coordinates):
    for g in consinks_df.keys():
        for cluster in consinks_df[g].index:
            consink_position = consinks_df[g].loc[cluster, 'position']

            for g2 in goal_coordinates.keys():
                distance_to_goal = np.sqrt((consink_position[0] - goal_coordinates[g2][0])**2 + (consink_position[1] - goal_coordinates[g2][1])**2)
                consinks_df[g].loc[cluster, f'distance_to_goal{g2}'] = distance_to_goal  
    
    return consinks_df


def consink_distance_stats(consinks_df):
    statistics = {}
    goals = list(consinks_df.keys())
    for g in goals:
        # data1 is the f'distance_to_goal{goals[0]}' column
        data1 = consinks_df[g][f'distance_to_goal{goals[0]}'].to_numpy()
        data2 = consinks_df[g][f'distance_to_goal{goals[1]}'].to_numpy()

        # calculate the wilcoxon rank sum test
        statistic, p = wilcoxon(data1, data2)
        statistics[g] = {'n': data1.shape[0], 'statistic': statistic, 'p': p}

    # make a dataframe from statistics with one row for each g in goals
    statistics = pd.DataFrame(statistics).T

    # add a column 'significant' to statistics, containing True if the p value is less than 0.05
    statistics['significant'] = statistics['p'] < 0.05

    # add the name "goal" for the first column
    statistics.index.name = 'goal'

    return statistics



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


def plot_consink_distances_to_goal(consinks_df, fig_path=None):

    if fig_path is None:
        # throw an error
        raise ValueError('path must be provided')

    # create a new dataframe for seaborn plotting
    data = []

    goals = list(consinks_df.keys())

    for g in goals:
        for cluster in consinks_df[g].index:
            for g2 in goals:
                distance_to_goal = consinks_df[g].loc[cluster, f'distance_to_goal{g2}'] * cm_per_pixel
                data.append({'unit': cluster, 'category': f'g{g}_d2g{g2}', 'distance': distance_to_goal})

    # convert to dataframe
    df = pd.DataFrame(data)

    # create hues for units
    # unique_values = df['unit'].unique()
    # palette = sns.color_palette("husl", len(unique_values))
    # color_map = dict(zip(unique_values, palette))

    # create a figure without subplots
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    # plot the swarmplot
    # ax = sns.swarmplot(data=df, x='category', y='distance', hue="unit", palette=color_map)
    ax = sns.swarmplot(data=df, x='category', y='distance')
    # ax.legend_.remove()
    # make y label "distance (cm)"
    ax.set_ylabel('distance (cm)')

    plt.tight_layout()
    plt.show()

    # save the figure
    fig.savefig(f'{fig_path}.png')
    fig.savefig(f'{fig_path}.svg')

    return fig


def load_consink_df(directory):
    consinks_df = load_pickle('consinks_df_translated_ctrl', directory)

    # save the consinks_df as a csv file
    for g in consinks_df.keys():
        consinks_df[g].to_csv(os.path.join(directory, f'consinks_goal_translated{g}.csv'))

    return consinks_df


# convert consink_df to an object that contains methods for restricing data by region and significance
class ConSinkData:
    def __init__(self, consinks_df, good_clusters):
        self.consinks_df = consinks_df

        self.consinks_df['cluster_id'] = consinks_df.index.str.replace('cluster_', '').astype(int)

        # add a region column to the consinks_df, where the regions is taken from the row in the good_clusters dataframe
        # with the same cluster_id
        good_clusters.set_index('cluster_id', inplace=True)
        cluster_to_region = good_clusters['region'].to_dict()
        self.consinks_df['region'] = consinks_df['cluster_id'].map(cluster_to_region)

    def print_consinks(self):
        print(ConSinkData.consinks_df.to_string())

    def restrict_by_region(self, region):
        """Restrict the DataFrame to a specific region."""
        restricted_df = {g: df[df['region'] == region] for g, df in self.consinks_df.items()}
        
        return restricted_df
    
    def restrict_by_significance(self):
        """Restrict the DataFrame to significant consinks."""
        restricted_df = {g: df[df['mrl'] > df['ci_95']] for g, df in self.consinks_df.items()}
        return restricted_df
    
    def restrict_by_region_and_significance(self, region):
        """Restrict the DataFrame to a specific region and significant consinks."""
        restricted_df = {g: df[(df['region'] == region) & (df['mrl'] > df['ci_95'])] for g, df in self.consinks_df.items()}
        return restricted_df




def main(experiment = 'robot_single_goal', animal = 'Rat_HC4', session = '01-08-2024', code_to_run = [9]):

    data_dir = get_data_dir(experiment, animal, session)    
    
    spike_dir = os.path.join(data_dir, 'spike_sorting')

    # get direction bins
    direction_bins = get_direction_bins(n_bins=12)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)
    dlc_data = calculate_frame_durations(dlc_data)
    dlc_data_concat = concatenate_dlc_data(dlc_data)
    save_pickle(dlc_data_concat, 'dlc_data_concat', dlc_dir)

    # get x and y limits
    limits = get_axes_limits(dlc_data_concat)

    # units = load_pickle('units_by_goal', spike_dir)
    units = load_pickle('units_w_behav_correlates', spike_dir)

    neuron_types = load_pickle('neuron_types', spike_dir)

    # goals = units.keys()

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
        
        consinks = {}
        consinks_df = {}
        # for goal in goals:
        #     goal_units = units[goal]
        #     consinks[goal] = {}
            
        #     for cluster in goal_units.keys():
        #         unit = concatenate_unit_across_trials(goal_units[cluster])
                
        #         # get consink  
        #         max_mrl, max_mrl_indices, mean_angle = find_consink(unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks)
        #         consink_position = np.round([candidate_sinks['x'][max_mrl_indices[1][0]], candidate_sinks['y'][max_mrl_indices[0][0]]], 3)
        #         consinks[goal][cluster] = {'mrl': max_mrl, 'position': consink_position, 'mean_angle': mean_angle}

        #     # create a data frame with the consink positions
        #     consinks_df[goal] = pd.DataFrame(consinks[goal]).T


        consinks = {}
        
        for cluster in units.keys():
            
            if neuron_types[cluster] == 'interneuron':
                continue

            unit = concatenate_unit_across_trials(units[cluster])
            
            # get consink  
            max_mrl, max_mrl_indices, mean_angle = find_consink(unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks)
            consink_position = np.round([candidate_sinks['x'][max_mrl_indices[1][0]], candidate_sinks['y'][max_mrl_indices[0][0]]], 3)
            consinks[cluster] = {'mrl': max_mrl, 'position': consink_position, 'mean_angle': mean_angle}

        # create a data frame with the consink positions
        consinks_df = pd.DataFrame(consinks).T
        
        # save consinks_df 
        save_pickle(consinks_df, 'consinks_df', spike_dir)
        # save as csv
        consinks_df.to_csv(os.path.join(spike_dir, 'consinks_df.csv'))


    ######################### PLOT CONSINKS FOR SINGLE GOAL EXPERIMENT ############################
    if 9 in code_to_run:
        # load consinks
        consinks_df = load_pickle('consinks_df_translated_ctrl', spike_dir)

        # load good_clusters.csv which has the brain regions for the clusters
        good_clusters = pd.read_csv(os.path.join(spike_dir, 'good_clusters.csv'), index_col=0)

        # create ConSinkData object
        consinks = ConSinkData(consinks_df, good_clusters)

        # get goal coordinates
        goal_coordinates = get_goal_coordinates(data_dir=data_dir)        

        # add a cluster_id column to the consinks_df which removes the 'cluster_' from the index and converts it to an int
        # consinks_df['cluster_id'] = consinks_df.index.str.replace('cluster_', '').astype(int)

        # add a region column to the consinks_df, where the regions is taken from the row in the good_clusters dataframe
        # with the same cluster_id
        # good_clusters.set_index('cluster_id', inplace=True)
        # cluster_to_region = good_clusters['region'].to_dict()
        # consinks_df['region'] = consinks_df['cluster_id'].map(cluster_to_region)

        # separate the consinks_df by brain region
        regions = ['CA1', 'CA3-DG']

        # make folder consinks in spike_dir if it doesn't already exist
        plot_dir = os.path.join(spike_dir, 'consinks')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # calculate a jitter amount to jitter the positions by so they are visible
        x_diff = np.mean(np.diff(candidate_sinks['x']))
        y_diff = np.mean(np.diff(candidate_sinks['y']))
        jitter = (x_diff/3, y_diff/3)

        for region in regions:
            # deep copy the consinks_df, keeping only the rows with the region
            # consinks_df_region = consinks_df[consinks_df['region'] == region].copy()
            consinks_df_region = consinks.restrict_by_region_and_significance(region)

            plot_all_consinks(consinks_df_region, goal_coordinates, limits, jitter, plot_dir, plot_name=f'ConSinks_{region}')

            
    pass

       
    
    # ######################### TEST STATISTICAL SIGNIFICANCE OF CONSINKS #########################
    # # shift the head directions relative to their positions, and recalculate the tuning to the 
    # # previously identified consink position. 
    # if 1 in code_to_run:
    #     # load the consinks_df
    #     consinks_df = load_pickle('consinks_df', spike_dir)
    #     # add two columns to hold the confidence intervals

    #     for goal in goals:
    #         goal_units = units[goal]
    #         # consinks[goal] = {}
            
    #         # make columns for the confidence intervals; place them directly beside the mrl column
    #         idx = consinks_df[goal].columns.get_loc('mrl')

    #         # if the columns don't exist, insert them            
    #         if 'ci_95' not in consinks_df[goal].columns:
    #             consinks_df[goal].insert(idx + 1, 'ci_95', np.nan)
    #             consinks_df[goal].insert(idx + 2, 'ci_999', np.nan)

    #         for cluster in goal_units.keys():
    #             unit = concatenate_unit_across_trials(goal_units[cluster])

    #             candidate_sink = consinks_df[goal].loc[cluster, 'position']
    #             # find the indices of the candidate sink in the candidate_sinks dictionaries
    #             sink_x_index = np.where(np.round(candidate_sinks['x'], 3) == candidate_sink[0])[0][0]
    #             sink_y_index = np.where(np.round(candidate_sinks['y'], 3) == candidate_sink[1])[0][0]

    #             # reldir_occ_by_pos_4sink = reldir_occ_by_pos[:, :, sink_y_index, sink_x_index, :]

    #             print(f'calcualting confidence intervals for goal {goal} {cluster}')
    #             # ci = recalculate_consink_to_single_candidate_from_shuffle(unit, reldir_occ_by_pos_4sink, candidate_sink, direction_bins)
    #             ci = recalculate_consink_to_all_candidates_from_shuffle(unit, reldir_occ_by_pos, sink_bins, candidate_sinks)
                
    #             consinks_df[goal].loc[cluster, 'ci_95'] = ci[0]
    #             consinks_df[goal].loc[cluster, 'ci_999'] = ci[1]

    #     save_pickle(consinks_df, 'consinks_df', spike_dir)

    ########## PLOT ALL SIGNIFICANT CONSINKS ####################################
    if 2 in code_to_run:

        # get goal coordinates
        goal_coordinates = get_goal_coordinates(data_dir=data_dir)

        # make folder consinks in spike_dir if it doesn't already exist
        plot_dir = os.path.join(spike_dir, 'consinks')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        
        # load the consinks_df
        consinks_df = load_pickle('consinks_df_translated_ctrl', spike_dir)

        # calculate a jitter amount to jitter the positions by so they are visible
        x_diff = np.mean(np.diff(candidate_sinks['x']))
        y_diff = np.mean(np.diff(candidate_sinks['y']))
        jitter = (x_diff/3, y_diff/3)
        plot_all_consinks(consinks_df, goal_coordinates, limits, jitter, plot_dir)
        

    # ################ CALCULATE CONSINK DISTANCE TO GOAL #######Finsbury Park#######################
    # if 3 in code_to_run:Finsbury Park
    #     goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    #     # get the distance to goal for each consink
    #     consinks_df = load_pickle('consinks_df', spike_dir)
    #     consinks_df = restrict_to_significant_consinks(consinks_df)
    #     consinks_df = calculate_consink_distance_to_goal(consinks_df, goal_coordinates)

    #     # calculate the statistics for the differences in distance to goal for each consink
    #     statistics = consink_distance_stats(consinks_df)
    #     # save statistics as csv file
    #     statistics.to_csv(os.path.join(spike_dir, 'consinks', 'consink_distance_statistics.csv'))

    #     ############ plot consink distances to goal ############################
    #     fig_path = os.path.join(spike_dir, 'consinks', 'consink_distances_to_goal_swarmplot')
    #     plot_consink_distances_to_goal(consinks_df, fig_path=fig_path)


def main2(experiment='robot_single_goal', animal='Rat_HC2', session='16-07-2024'):

    data_dir = get_data_dir(experiment, animal, session)    
    
    spike_dir = os.path.join(data_dir, 'spike_sorting')

    # get direction bins
    direction_bins = get_direction_bins(n_bins=12)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    # dlc_data = load_pickle('dlc_final', dlc_dir)
    dlc_data_concat = load_pickle('dlc_data_concat_by_choice', dlc_dir)
    # dlc_data = calculate_frame_durations(dlc_data)
    # dlc_data_concat = concatenate_dlc_data(dlc_data)
    # save_pickle(dlc_data_concat, 'dlc_data_concat', dlc_dir)

    # get x and y limits
    limits = get_axes_limits(dlc_data_concat['correct'])

    # units = load_pickle('units_by_goal', spike_dir)
    # units = load_pickle('units_w_behav_correlates', spike_dir)
    units = load_pickle('units_concat_by_choice', spike_dir)

    neuron_types = load_pickle('neuron_types', spike_dir)

    # goals = units.keys()

    choice_types = ['correct', 'incorrect']

    reldir_occ_by_pos = {}
    sink_bins = {}
    candidate_sinks = {}

    for c in choice_types:
    # get relative direction occupancy by position if np array not already saved
        file_name = f'reldir_occ_by_pos_{c}.npy'
        if os.path.exists(os.path.join(dlc_dir, file_name)) == False:
            reldir_occ_by_pos[c], sink_bins[c], candidate_sinks[c] = get_relative_direction_occupancy_by_position(dlc_data_concat[c], limits)
            np.save(os.path.join(dlc_dir, file_name), reldir_occ_by_pos[c])
            # save sink bins and candidate sinks as pickle files
            save_pickle(sink_bins[c], f'sink_bins_{c}', dlc_dir)
            save_pickle(candidate_sinks[c], f'candidate_sinks_{c}', dlc_dir)     

        else:
            reldir_occ_by_pos[c] = np.load(os.path.join(dlc_dir, file_name))
            sink_bins[c] = load_pickle(f'sink_bins_{c}', dlc_dir)
            candidate_sinks[c] = load_pickle(f'candidate_sinks_{c}', dlc_dir)

    ################# CALCULATE CONSINKS ###########################################     
    consinks = {}
    consinks_df = {}
    
    for c in choice_types:
        consinks[c] = {}
        for cluster in units.keys():
            
            if neuron_types[cluster] == 'interneuron':
                continue

            unit = concatenate_unit_across_trials(units[cluster][c])
            
            # check if unit is empty df
            if unit.empty:
                consinks[c][cluster] = {'mrl': np.nan, 'position': np.nan, 'mean_angle': np.nan}
                continue

            # get consink  
            max_mrl, max_mrl_indices, mean_angle = find_consink(unit, reldir_occ_by_pos[c], sink_bins[c], direction_bins, candidate_sinks[c])
            consink_position = np.round([candidate_sinks[c]['x'][max_mrl_indices[1][0]], candidate_sinks[c]['y'][max_mrl_indices[0][0]]], 3)
            consinks[c][cluster] = {'mrl': max_mrl, 'position': consink_position, 'mean_angle': mean_angle}

        # create a data frame with the consink positions
        consinks_df[c] = pd.DataFrame(consinks[c]).T
        
        
        # save as csv            
        consinks_df[c].to_csv(os.path.join(spike_dir, f'consinks_df_{c}.csv'))

    # save consinks_df 
    save_pickle(consinks_df, 'consinks_df_by_choice', spike_dir)

        
if __name__ == "__main__":
    
    # main()
    main2(experiment='robot_single_goal', animal='Rat_HC2', session='16-07-2024')


