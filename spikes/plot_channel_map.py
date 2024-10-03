import os 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ast

import sys
# if on windows system
if os.name == 'nt':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')

# if on linux system
else:
    sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')

from utilities.get_directories import get_data_dir 


def plot_clusters_on_probe_V1(channel_positions, cluster_info, spike_dir):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # plot channel positions in blue
    plt.plot(channel_positions[:,0], channel_positions[:,1], 'bo')

    # get the indices of all values in the group column of the cluster_info dataframe that are "good"
    good_clusters = cluster_info[cluster_info['group'] == 'good']

    # get the positions of the good channels
    good_channel_positions = channel_positions[good_clusters['ch'], :]

    # plot the good channels in red
    plt.plot(good_channel_positions[:,0], good_channel_positions[:,1], 'ro')

    # save the plot the spike directory
    plt.savefig(os.path.join(spike_dir, 'clusters_plotted_on_probe.png'))

def plot_clusters_on_probe(channel_positions, cluster_info, spike_dir):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # plot channel positions in blue
    plt.plot(channel_positions[:,0], channel_positions[:,1], 'bo')

    # get the indices of all values in the group column of the cluster_info dataframe that are "good"
    # good_clusters = cluster_info['group'] == 'good'
    good_clusters = cluster_info[cluster_info['group'] == 'good']

    # use these indices to find the channels
    # good_channels = cluster_info['ch'].loc[good_clusters]

    # get the positions of the good channels
    # good_channel_positions = channel_positions[good_channels, :]
    good_channel_positions = channel_positions[good_clusters['ch']]

    # plot the good channels in red
    plt.plot(good_channel_positions[:,0], good_channel_positions[:,1], 'ro')

    # save the plot the spike directory
    plt.savefig(os.path.join(spike_dir, 'clusters_plotted_on_probe.png'))

    return good_clusters


def probe_positions_by_hpc_region_old(animal, session):
    # open the cluster_plotted_on_probe.png file and manually enter the shank number and shank depth for each region

    regions = "CA1", "CA3-DG"
    shank_numbers = [1, 2, 3, 4]

    # create a csv file with regions as the rows and shank numbers as the columns
    # enter the shank depth for each region and shank number
    region_depth = pd.DataFrame(index=regions, columns=shank_numbers)

    print(f"animal = {animal}, session = {session}")

    for region in regions:
        for shank in shank_numbers:
            units_in_region = input('are there any ' + region + ' units on shank ' + str(shank) + '? (y/n)')
            if units_in_region == 'y':
                # shank_depth is a 2 element tuple with the first element being the lower tip distance and the second element being the upper tip distance 
                lower_tip_distance = int(input('what is the lower tip distance of the ' + region + ' units on shank ' + str(shank) + '?'))
                upper_tip_distance = int(input('what is the upper tip distance of the ' + region + ' units on shank ' + str(shank) + '?'))
                region_depth.loc[region, shank] = (lower_tip_distance, upper_tip_distance)

    return region_depth


def probe_positions_by_hpc_region(animal, session):
    # open the cluster_plotted_on_probe.png file and manually enter the shank number and shank depth for each region

    regions = "CA1", "CA3-DG"

    # create a csv file with regions as the rows and depth as the columns
    region_depth = pd.DataFrame(index=regions, columns=["depth"], dtype=object)

    print(f"animal = {animal}, session = {session}")

    for region in regions:
        
        units_in_region = input(f'are there any {region} units? (y/n)')
        if units_in_region == 'y':
            # shank_depth is a 2 element tuple with the first element being the lower tip distance and the second element being the upper tip distance 
            lower_tip_distance = int(input('what is the lower tip distance of the ' + region + ' units on the probe?'))
            upper_tip_distance = int(input('what is the upper tip distance of the ' + region + ' units on the probe?'))
            region_depth.loc[region, "depth"] = (lower_tip_distance, upper_tip_distance)

    return region_depth


def separate_clusters_by_region_and_shank(good_clusters, channel_positions, region_depth, spike_dir):

    regions = region_depth.index.tolist()

    shank_x_positions = {1: (0, 32), 2: (250, 282), 3: (500, 532), 4: (750, 782)}

    channel_positions.shape

    # make 2 new columns in the good_clusters dataframe for the shank number and the brain region
    good_clusters_new = good_clusters.copy()
    good_clusters_new.loc[:, 'shank'] = -1
    good_clusters_new.loc[:, 'region'] = ''

    for cluster in good_clusters_new.iterrows():
        cluster_position = channel_positions[cluster[1]['ch'], :]
        for shank, x_positions in shank_x_positions.items():
            if cluster_position[0] == x_positions[0] or cluster_position[0] == x_positions[1]:
                good_clusters_new.loc[cluster[0], 'shank'] = int(shank)
                break
        
        for region in regions:
            if pd.notna(region_depth.loc[region, 'depth']):
                depth = ast.literal_eval(region_depth.loc[region, 'depth'])
                lower_tip_distance = depth[0]
                upper_tip_distance = depth[1]
                if cluster_position[1] >= lower_tip_distance and cluster_position[1] <= upper_tip_distance:
                    good_clusters_new.loc[cluster[0], 'region'] = region
                    break

    good_clusters_new.to_csv(os.path.join(spike_dir, 'good_clusters.csv'))
    return good_clusters_new



def main(experiment = 'robot_single_goal', animal = 'Rat_HC4', session = '01-08-2024'):

    data_dir = get_data_dir(experiment, animal, session)    
    
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    spike_sorting_dir = os.path.join(spike_dir, 'sorter_output')

    map_file = "channel_map.npy"
    
    channel_map = np.load(os.path.join(spike_sorting_dir, map_file))

    channel_positions_file = "channel_positions.npy"
    channel_positions = np.load(os.path.join(spike_sorting_dir, channel_positions_file))

    spike_clusters = np.load(os.path.join(spike_sorting_dir, 'spike_clusters.npy'))
    
    cluster_quality = pd.read_csv(os.path.join(spike_sorting_dir, 
                           'cluster_group.tsv'), sep='\t')
    
   
    cluster_info = pd.read_csv(os.path.join(spike_sorting_dir, 'cluster_info.tsv'), sep='\t')

    good_clusters = plot_clusters_on_probe(channel_positions, cluster_info, spike_dir)

    region_depth = probe_positions_by_hpc_region(animal, session)
    region_depth.to_csv(os.path.join(spike_dir, 'region_depth.csv'))
    del region_depth
    # load region_depth from csv file
    region_depth = pd.read_csv(os.path.join(spike_dir, 'region_depth.csv'))
    # remove the index column, and make the first column the index
    region_depth = region_depth.set_index('Unnamed: 0')

    good_clusters = separate_clusters_by_region_and_shank(good_clusters, channel_positions, region_depth, spike_dir)  

    pass



if __name__ == "__main__":
    
    main()

    