import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt


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
    plot_spikes_2goals(units_by_goal, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)
        
    pass