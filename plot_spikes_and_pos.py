import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from scipy import ndimage


from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from load_behaviour import split_dictionary_by_goal
from calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, cm_per_pixel


def basic_spike_pos_plot(ax, unit, dlc_data, goal_coordinates, x_and_y_limits):
    
    # plot the goal positions
    colours = ['b', 'g']
    for i, g in enumerate(goal_coordinates.keys()):
        # draw a circle with radius 80 around the goal on ax
        circle = plt.Circle((goal_coordinates[g][0], 
            goal_coordinates[g][1]), 80, color=colours[i], 
            fill=False, linewidth=10)
        ax.add_artist(circle)       
    
    for t in unit.keys():
        # plot every 10th position from the dlc data
        ax.plot(dlc_data[t]['x'][::10], dlc_data[t]['y'][::10], 'k.', markersize=4)

    # plot the spike positions
    for t in unit.keys():
        ax.plot(unit[t]['x'], unit[t]['y'], 'r.', markersize=1)
    
    ax.set_xlabel('x (cm)')
    ax.set_ylabel('y (cm)')

    # set the x and y limits
    ax.set_xlim([x_and_y_limits['x'][0] - 50, x_and_y_limits['x'][1] + 50])
    ax.set_ylim(x_and_y_limits['y'][0] - 50, x_and_y_limits['y'][1] + 50)

    # convert tick labels to cm
    xticks = ax.get_xticks()
    xticks = np.int32(np.round(xticks * cm_per_pixel, 0))
    ax.set_xticklabels(xticks)

    yticks = ax.get_yticks()
    yticks = np.int32(np.round(yticks * cm_per_pixel, 0))
    ax.set_yticklabels(yticks)

    # flip the y axis so that it matches the video
    ax.invert_yaxis()

    ax.set_aspect('equal', 'box')

def plot_spikes_and_pos(units, dlc_data, goal_coordinates, x_and_y_limits, plot_dir):

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    for u in units.keys():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(u)

        basic_spike_pos_plot(ax, units[u], dlc_data, goal_coordinates, x_and_y_limits)
        
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        plt.close(fig)

def plot_spikes_2goals(units_by_goal, dlc_data, goal_coordinates, x_and_y_limits, plot_dir):

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    for u in units_by_goal.keys():
        # figure will have 2 subplots, one for each goal
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))

        for i, g in enumerate(units_by_goal[u].keys()):

            ax[i].set_title(f'{u} - goal{g}')
            
            basic_spike_pos_plot(ax[i], units_by_goal[u][g], dlc_data, {g: goal_coordinates[g]}, x_and_y_limits)
        
        # plt.show()
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        plt.close(fig)


def plot_rate_maps(rate_maps, plot_dir):
    # plot the rate maps
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    x_bins = rate_maps['x_bins']
    y_bins = rate_maps['y_bins']

    occupancy = rate_maps['occupancy']

    for u in rate_maps['rate_maps'].keys():
        rate_map = rate_maps['rate_maps'][u]
        rate_map_copy = rate_map.copy()

        # make any bin with occupancy less than 1 into nan
        rate_map_copy[occupancy < 1] = np.nan

        # get masked array, which tells us where the nans are
        masked_array = np.ma.masked_invalid(rate_map_copy)

        # make all the nans the average of the surrounding non-nan values
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
        rate_map_smoothed = np.ma.filled(masked_array, fill_value=np.NaN)

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(u)

        ax.imshow(rate_map_smoothed, cmap='jet', aspect='auto')
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
        # don't need to flip the y axis because it's an image, so plots from top dow
        ax.set_aspect('equal', 'box')

        # get x_ticks
        xticks = ax.get_xticks()
        # set the x ticks so that only those that are between 0 and n_x_bins are shown
        xticks = xticks[(xticks >= 0) & (xticks < len(x_bins))]
        ax.set_xticks(xticks)
        # interpolate the x values to get the pixel values, noting that 0.5 needs to be added to the xticks, because they are centred on their bins
        xtick_values = np.int32(np.round(np.interp(xticks + 0.5, np.arange(len(x_bins)), x_bins), 0))
        # then convert to cm 
        xtick_values = np.int32(np.round(xtick_values * cm_per_pixel, 0))        
        ax.set_xticklabels(xtick_values)

        # do the same for y_ticks
        yticks = ax.get_yticks()
        yticks = yticks[(yticks >= 0) & (yticks < len(y_bins))]
        ax.set_yticks(yticks)
        ytick_values = np.int32(np.round(np.interp(yticks + 0.5, np.arange(len(y_bins)), y_bins), 0))
        ytick_values = np.int32(np.round(ytick_values * cm_per_pixel, 0))
        ax.set_yticklabels(ytick_values)

        # show the plot
        plt.show()
        
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))

        plt.close(fig)


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # get goal coordinates
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # get x and y limits
    x_and_y_limits = get_x_and_y_limits(dlc_data)

    # load the positional occupancy data
    positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

    # load the directional occupancy data
    directional_occupancy = load_pickle('directional_occupancy', dlc_dir)

    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_w_behav_correlates', spike_dir)

    # plot spikes and position
    plot_dir = os.path.join(spike_dir, 'spikes_and_pos')
    # plot_spikes_and_pos(units, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)

    # plot spike and position by goal
    units_by_goal = {}
    for u in units.keys():
        units_by_goal[u] = split_dictionary_by_goal(units[u], data_dir)
    
    plot_dir = os.path.join(spike_dir, 'spikes_and_pos_by_goal')
    # plot_spikes_2goals(units_by_goal, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)

    # plot rate maps
    plot_dir = os.path.join(spike_dir, 'rate_maps')
    rate_maps = load_pickle('rate_maps', spike_dir)

    plot_rate_maps(rate_maps, plot_dir)

    

        
    pass