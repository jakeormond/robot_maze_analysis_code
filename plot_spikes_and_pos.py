import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


from get_directories import get_data_dir, get_robot_maze_directory

def plot_spikes_and_pos(units, dlc_data, spike_dir):

    plot_dir = os.path.join(spike_dir, 'spikes_and_pos')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    for u in units.keys():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        for t in units[u].keys():
            # plot every 10th position from the dlc data
            ax.plot(dlc_data[t]['x'][::10], dlc_data[t]['y'][::10], 'k.', markersize=4)

        # plot the spike positions
        for t in units[u].keys():
            ax.plot(units[u][t]['x'], units[u][t]['y'], 'r.', markersize=1)
        
        ax.set_title(u)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # flip the y axis so that it matches the video
        ax.invert_yaxis()
        # plt.show()
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        plt.close(fig)



if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_final.pkl')
    with open(dlc_pickle_path, 'rb') as f:
        dlc_data = pickle.load(f)

    # load the positional occupancy data
    positional_occupancy_file = os.path.join(dlc_dir, 'positional_occupancy.pkl')
    with open(positional_occupancy_file, 'rb') as f:
        positional_occupancy = pickle.load(f)

    # load the directional occupancy data
    directional_occupancy_file = os.path.join(dlc_dir, 'directional_occupancy.pkl')
    with open(directional_occupancy_file, 'rb') as f:
        directional_occupancy = pickle.load(f)

    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units_file = os.path.join(spike_dir, 'units_w_behav_correlates.pickle')
    with open(units_file, 'rb') as handle:
        units = pickle.load(handle)

    # plot spikes and position
    plot_spikes_and_pos(units, dlc_data, spike_dir)