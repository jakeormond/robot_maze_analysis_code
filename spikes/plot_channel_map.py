import os 
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir 


def plot_clusters_on_probe(channel_positions, cluster_info, spike_dir):

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # plot channel positions in blue
    plt.plot(channel_positions[:,0], channel_positions[:,1], 'bo')

    # get the indices of all values in the group column of the cluster_info dataframe that are "good"
    good_clusters = cluster_info['group'] == 'good'

    # use these indices to find the channels
    good_channels = cluster_info['ch'].loc[good_clusters]

    # get the positions of the good channels
    good_channel_positions = channel_positions[good_channels, :]

    # plot the good channels in red
    plt.plot(good_channel_positions[:,0], good_channel_positions[:,1], 'ro')

    # save the plot the spike directory
    plt.savefig(os.path.join(spike_dir, 'clusters_plotted_on_probe.png'))



if __name__ == "__main__":
    
    animal = 'Rat47'
    session = '16-02-2024'
    data_dir = get_data_dir(animal, session)

    spike_dir = os.path.join(data_dir, 'spike_sorting')
    map_file = "channel_map.npy"
    channel_map = np.load(os.path.join(spike_dir, map_file))

    channel_positions_file = "channel_positions.npy"
    channel_positions = np.load(os.path.join(spike_dir, channel_positions_file))

    spike_clusters = np.load(os.path.join(spike_dir, 'spike_clusters.npy'))
    
    cluster_quality = pd.read_csv(os.path.join(spike_dir, 
                            'cluster_group.tsv'), sep='\t')
    
    cluster_info = pd.read_csv(os.path.join(spike_dir, 'cluster_info.tsv'), sep='\t')


    plot_clusters_on_probe(channel_positions, cluster_info, spike_dir)
    pass