import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from get_directories import get_data_dir, get_robot_maze_directory
from calculate_pos_and_dir import get_goal_coordinates
from load_and_save_data import load_pickle, save_pickle


def plot_occupancy_heatmap(positional_occupancy, figure_dir):
    
    x_bins = positional_occupancy['x_bins']
    y_bins = positional_occupancy['y_bins']

    n_x_bins = x_bins.shape[0]
    n_y_bins = y_bins.shape[0]

    # set ticks at every 5th bin
    x_tick_positions = np.arange(0, n_x_bins, 5) 
    y_tick_positions = np.arange(0, n_y_bins, 5)    
    
    # plot the positional occupancy as a heatmap
    plt.figure()
    plt.imshow(positional_occupancy['occupancy'], cmap='hot')


    # convert the tick values to pixel values
    x_bins = np.linspace(x_bins[0], x_bins[-1], n_x_bins)
    y_bins = np.linspace(y_bins[0], y_bins[-1], n_y_bins)

    # Set the tick positions and labels
    plt.xticks(x_tick_positions - 0.5, np.int32(x_bins[x_tick_positions]))
    plt.yticks(y_tick_positions - 0.5, np.int32(y_bins[y_tick_positions]))

    plt.colorbar()
    plt.show()
    # save the figure
    figure_path = os.path.join(figure_dir, 'occupancy_heatmap.png')
    plt.savefig(figure_path)
    plt.close()


def plot_trial_path(dlc_data, limits, dlc_dir, d):
    # plot the trial path
    plt.figure()

    # first frames should be red, and transition to blue
    x_data = np.array(dlc_data['x'])
    y_data = np.array(dlc_data['y'])

    # keep only every 10th frame, since it is too slow otherwise
    x_data = x_data[::10]
    y_data = y_data[::10]

    n_frames = len(x_data)

    # create a color map
    cmap = plt.cm.get_cmap('coolwarm')
    # get the colors
    colors = cmap(np.linspace(0, 1, n_frames))
    # plot the trial path
    for i in range(n_frames):
        plt.scatter(x_data[i], y_data[i], color=colors[i])

    # set axes limits
    plt.xlim(limits['x_min'], limits['x_max'])
    plt.ylim(limits['y_min'], limits['y_max'])

    # flip the y axis
    plt.gca().invert_yaxis()

    plt.title(d)
    plt.show()

    # if trial_paths directory doesn't exist, create it
    trial_paths_dir = os.path.join(dlc_dir, 'trial_paths')
    if not os.path.exists(trial_paths_dir):
        os.makedirs(trial_paths_dir)
    
    # save the figure
    fig_path = os.path.join(trial_paths_dir, d + '.png')
    plt.savefig(fig_path)

    # close the figure
    plt.close()


def get_positional_occupancy(dlc_data, limits):    

    # NOTE THAT IN THE OUTPUT, THE FIRST INDEX IS THE Y AXIS, 
    # AND THE SECOND INDEX IS THE X AXIS

    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']
    x_width = limits['x_width']

    y_min = limits['y_min']
    y_max = limits['y_max']
    y_height = limits['y_height']

    # we want roughly 400 bins
    pixels_per_bin = np.sqrt(x_width*y_height/400)
    n_x_bins = int(np.round(x_width/pixels_per_bin)) # note that n_bins is actually one more than the number of bins
    n_y_bins = int(np.round(y_height/pixels_per_bin))

    # create bins
    x_bins_og = np.linspace(x_min, x_max, n_x_bins)
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin
    
    y_bins_og = np.linspace(y_min, y_max, n_y_bins)
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin

    # create positional occupancy matrix
    positional_occupancy = np.zeros((n_y_bins, n_x_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    x_bin = np.digitize(x, x_bins) - 1
    y_bin = np.digitize(y, y_bins) - 1 

    # loop through each frame and add to the appropriate bin
    for i in range(np.max(x_bin)):
        for j in range(np.max(y_bin)):
            # get the indices of the frames in the bin
            indices = np.where((x_bin==i) & (y_bin==j))[0]
            # add to the appropriate bin
            positional_occupancy[j, i] = np.sum(dlc_data['durations'][indices])

    x_and_y_bins = {'x_bins': x_bins_og, 'y_bins': y_bins_og}

    return positional_occupancy, x_and_y_bins


def concatenate_dlc_data(dlc_data):
    for i, d in enumerate(dlc_data.keys()):

        # calculate frame intervals
        times = dlc_data[d]['ts'].values
        frame_intervals = np.diff(times)
        # one less interval than frames, so we'll just replicate the last interval
        frame_intervals = np.append(frame_intervals, frame_intervals[-1])

        # add frame intervals to dlc_data
        frame_intervals = frame_intervals/1000 # convert to seconds
        # round to 4 decimal places, i.e. 0.1 ms
        frame_intervals = np.round(frame_intervals, 4)
        dlc_data[d]['durations'] = frame_intervals

        if i==0:
            dlc_data_concat = dlc_data[d]
        
        else:
            dlc_data_concat = pd.concat([dlc_data_concat, dlc_data[d]], 
                    ignore_index=True)            
   
    # get the x and y limits of the maze
    x_min = np.min(dlc_data_concat['x'])
    x_max = np.max(dlc_data_concat['x'])
    x_width = x_max - x_min

    y_min = np.min(dlc_data_concat['y'])
    y_max = np.max(dlc_data_concat['y'])    
    y_height = y_max - y_min

    limits =  {'x_min': x_min, 'x_max': x_max, 'x_width': x_width,
            'y_min': y_min, 'y_max': y_max, 'y_height': y_height}

    return dlc_data_concat, limits


def get_directional_occupancy(dlc_data):

    # create 24 bins, each 15 degrees
    n_bins = 24
    direction_bins_og = np.linspace(-np.pi, np.pi, n_bins+1)
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    # create list of dlc_data with directional data
    direction_data = {}
    direction_data['allocentric'] = []
    # start with column 'hd', and then find columns that begin 
    # with 'goal_direction' or 'screen_direction'
    for d in dlc_data.keys():
        if 'hd' in d:
            direction_data['allocentric'].append(d)
        elif 'goal_direction' in d:
            direction_data['allocentric'].append(d)
        elif 'screen_direction' in d:
            direction_data['allocentric'].append(d)       

    # create list of dlc_data with relative direction data
    direction_data['egocentric'] = []
    # find columns  that bign with 'relative_direction'
    for d in dlc_data.keys():
        if 'relative_direction' in d:
            direction_data['egocentric'].append(d)

    # directional_occupancy is a dictionary with keys 'allocentric' and 'egocentric'
    directional_occupancy = {'allocentric': {}, 'egocentric': {}}
    for direction_type in direction_data.keys():
        for d in direction_data[direction_type]:
            # bin_counts, bin_edges = \
            #     np.histogram(dlc_data[d], direction_bins)
            bin_indices = np.digitize(dlc_data[d], direction_bins, right=True) - 1
            # any bin_indices that are -1 should be 0
            bin_indices[bin_indices==-1] = 0
            # any bin_indices that are n_bins should be n_bins-1
            bin_indices[bin_indices==n_bins] = n_bins-1

            # get the occupancy for each bin, so a vector of length n_bins
            occupancy = np.zeros(n_bins)
            for i in range(n_bins):
                occupancy[i] = np.sum(dlc_data['durations'][bin_indices==i])

            directional_occupancy[direction_type][d] = occupancy

    return directional_occupancy, direction_bins_og


def plot_directional_occupancy(directional_occupancy, figure_dir):
    
    # note that polar plot converts radian to degrees. 0 degrees = 0 radians,
    # 90 degrees = pi/2 radians, 180 degrees = +/- pi radians, etc.

    # plot the directional occupancy
    bins = directional_occupancy['bins']
    # polar plot ticks are the centres of the bins
    tick_positions = np.round(bins[:-1] + np.diff(bins)/2, 2)
    tick_positions = np.append(tick_positions, tick_positions[0])
    
    for direction_type in directional_occupancy['occupancy'].keys():
        for d in directional_occupancy['occupancy'][direction_type].keys():
            plt.figure()

            occupancy = directional_occupancy['occupancy'][direction_type][d]
            # add concatenate the first value to the end so that the plot is closed
            occupancy = np.append(occupancy, occupancy[0])

            plt.polar(tick_positions, occupancy)

            plt.title(d)
            plt.show()
            
            # save the figure
            fig_path = os.path.join(figure_dir, f'{direction_type}_{d}.png')
            plt.savefig(fig_path)
            plt.close()


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load dlc_data which has the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)
    
    # load the platform coordinates, from which we can get the goal coordinates
    robot_maze_dir = get_robot_maze_directory()
    platform_dir = os.path.join(robot_maze_dir, 'workstation', 'map_files')
    platform_coordinates = load_pickle('platform_coordinates', platform_dir)

    # get goal coordinates 
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    # concatenate dlc_data
    dlc_data_concat, limits = concatenate_dlc_data(dlc_data)

    # calculate positional occupancy
    positional_occupancy_temp, x_and_y_bins = get_positional_occupancy(dlc_data_concat, limits)
    positional_occupancy = {'occupancy': positional_occupancy_temp, 'x_bins': x_and_y_bins['x_bins'], 'y_bins': x_and_y_bins['y_bins']}
    # save the positional_occupancy
    positional_occupancy_file = os.path.join(dlc_dir, 'positional_occupancy')

    with open(positional_occupancy_file, 'rb') as f:
        positional_occupancy = pickle.load(f)       
    
    # plot the the trial paths
    # for d in dlc_data.keys():
    #     plot_trial_path(dlc_data[d], limits, dlc_dir, d)

    # plot the heat map of occupancy
    plot_occupancy_heatmap(positional_occupancy, dlc_dir)
    
    # calculate directional occupancy
    # directional_occupancy_temp, direction_bins_og = get_directional_occupancy(dlc_data_concat)  
    # directional_occupancy = {'occupancy': directional_occupancy_temp, 'bins': direction_bins_og} 
    directional_occupancy_file = os.path.join(dlc_dir, 'directional_occupancy.pkl')
    # with open(directional_occupancy_file, 'wb') as f:
    # #     pickle.dump(directional_occupancy, f)

    # del directional_occupancy
    with open(directional_occupancy_file, 'rb') as f:
        directional_occupancy = pickle.load(f)

    # plot the directional occupancy
    figure_dir = os.path.join(dlc_dir, 'directional_occupancy_plots')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plot_directional_occupancy(directional_occupancy, figure_dir)
   
