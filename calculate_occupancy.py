import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from get_directories import get_data_dir, get_robot_maze_directory
from calculate_pos_and_dir import get_goal_coordinates, cm_per_pixel
from load_and_save_data import load_pickle, save_pickle
from load_behaviour import get_behaviour_dir


def plot_occupancy_heatmap(positional_occupancy, goal_coordinates, figure_dir):

    # if figure_dir doesn't exist, create it
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    
    x_bins = positional_occupancy['x_bins']
    y_bins = positional_occupancy['y_bins']

    # n_x_bins = x_bins.shape[0]
    # n_y_bins = y_bins.shape[0]

    # set ticks at every 5th bin
    # x_tick_positions = np.arange(0, n_x_bins, 5) 
    # y_tick_positions = np.arange(0, n_y_bins, 5)    
    
    # plot the positional occupancy as a heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = plt.imshow(positional_occupancy['occupancy'], cmap='hot')

    # plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    cbar.set_label('Occupancy (sec)', size=15)         
    cbar.ax.tick_params(labelsize=15)   

    ax.set_xlabel('x (cm)',fontsize=15)
    ax.set_ylabel('y (cm)', fontsize=15)
    # don't need to flip the y axis because it's an image, so plots from top dow
    ax.set_aspect('equal', 'box')

    # draw the goal locations
    colours = ['k', '0.5']
    for i, g in enumerate(goal_coordinates.keys()):
        # first, convert to heat map coordinates
        goal_x, goal_y = goal_coordinates[g]

        # Convert to heat map coordinates
        goal_x_heatmap = np.interp(goal_x, x_bins, np.arange(len(x_bins))) - 0.5
        goal_y_heatmap = np.interp(goal_y, y_bins, np.arange(len(y_bins))) - 0.5                          
        
        # draw a circle with radius 80 around the goal on ax
        circle = plt.Circle((goal_x_heatmap, 
            goal_y_heatmap), radius=1, color=colours[i], 
            fill=False, linewidth=4)
        ax.add_artist(circle)

    #  get x_ticks
    xticks = ax.get_xticks()
    # set the x ticks so that only those that are between 0 and n_x_bins are shown
    xticks = xticks[(xticks >= 0) & (xticks < len(x_bins))]
    ax.set_xticks(xticks)
    # interpolate the x values to get the pixel values, noting that 0.5 needs to be added to the xticks, because they are centred on their bins
    xtick_values = np.int32(np.round(np.interp(xticks + 0.5, np.arange(len(x_bins)), x_bins), 0))
    # then convert to cm 
    xtick_values = np.int32(np.round(xtick_values * cm_per_pixel, 0))        
    ax.set_xticklabels(xtick_values)
    ax.tick_params(axis='x', labelsize=15)

    # do the same for y_ticks
    yticks = ax.get_yticks()
    yticks = yticks[(yticks >= 0) & (yticks < len(y_bins))]
    ax.set_yticks(yticks)
    ytick_values = np.int32(np.round(np.interp(yticks + 0.5, np.arange(len(y_bins)), y_bins), 0))
    ytick_values = np.int32(np.round(ytick_values * cm_per_pixel, 0))
    ax.set_yticklabels(ytick_values)
    ax.tick_params(axis='y', labelsize=15)

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


def get_directional_occupancy_by_position(dlc_data, limits):

    # NOTE THAT IN THE OUTPUT, THE FIRST INDEX IS THE Y AXIS, 
    # AND THE SECOND INDEX IS THE X AXIS

    # get the x and y limits of the maze
    x_min = limits['x_min']
    x_max = limits['x_max']
    x_width = limits['x_width']

    y_min = limits['y_min']
    y_max = limits['y_max']
    y_height = limits['y_height']

    # we want roughly 100 bins
    n_bins = 100
    pixels_per_bin = np.sqrt(x_width*y_height/n_bins)
    n_x_bins = int(np.round(x_width/pixels_per_bin)) # note that n_bins is actually one more than the number of bins
    n_y_bins = int(np.round(y_height/pixels_per_bin))

    # create bins
    x_bins_og = np.linspace(x_min, x_max, n_x_bins + 1)
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin
    
    y_bins_og = np.linspace(y_min, y_max, n_y_bins + 1)
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin

    # create directional occupancy by position array
    n_dir_bins=12 # 12 bins of 30 degrees each
    directional_occupancy_temp = np.zeros((n_y_bins, n_x_bins, n_dir_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    x_bin = np.digitize(x, x_bins) - 1
    y_bin = np.digitize(y, y_bins) - 1 

    # get the head direction
    hd = dlc_data['hd']

    # get the durations
    durations = dlc_data['durations']

    for i in range(np.max(x_bin)+1):
        for j in range(np.max(y_bin)+1):
            # get the indices where x_bin == i and y_bin == j
            indices = np.where((x_bin == i) & (y_bin == j))[0]

            # get the head directions and durations for these indices
            hd_temp = hd[indices]
            durations_temp = durations[indices]

            # get the directional occupancy for these indices
            directional_occupancy, direction_bins = \
                get_directional_occupancy(hd_temp, durations_temp, n_bins=n_dir_bins)

            # add the directional occupancy to positional_occupancy_temp
            directional_occupancy_temp[j, i, :] = directional_occupancy

    directional_occupancy_temp = np.round(directional_occupancy_temp, 3)

    directional_occupancy_by_position = {'occupancy': directional_occupancy_temp, 
                            'x_bins': x_bins_og, 'y_bins': y_bins_og, 'direction_bins': direction_bins}

    return directional_occupancy_by_position


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
    x_bins_og = np.linspace(x_min, x_max, n_x_bins + 1)
    x_bins = x_bins_og.copy()
    x_bins[-1] = x_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin
    
    y_bins_og = np.linspace(y_min, y_max, n_y_bins + 1)
    y_bins = y_bins_og.copy()
    y_bins[-1] = y_bins[-1] + 1e-6 # add a small number to the last bin so that the last value is included in the bin

    # create positional occupancy matrix
    positional_occupancy_temp = np.zeros((n_y_bins, n_x_bins))

    # get x and y data 
    x = dlc_data['x']
    y = dlc_data['y']

    x_bin = np.digitize(x, x_bins) - 1
    y_bin = np.digitize(y, y_bins) - 1 

    # sort the x and y bins into the positional occupancy matrix
    for i, (x_ind, y_ind) in enumerate(zip(x_bin, y_bin)):        
        positional_occupancy_temp[y_ind, x_ind] += dlc_data['durations'][i]
        
    positional_occupancy_temp = np.round(positional_occupancy_temp, 3)

    positional_occupancy = {'occupancy': positional_occupancy_temp, 
                            'x_bins': x_bins_og, 'y_bins': y_bins_og}

    return positional_occupancy


def calculate_frame_durations(dlc_data):
    for d in dlc_data.keys():

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

    return dlc_data


def get_axes_limits(dlc_data):
    # get the x and y limits of the maze
    x_min = np.min(dlc_data['x'])
    x_max = np.max(dlc_data['x'])
    x_width = x_max - x_min

    y_min = np.min(dlc_data['y'])
    y_max = np.max(dlc_data['y'])    
    y_height = y_max - y_min

    limits =  {'x_min': x_min, 'x_max': x_max, 'x_width': x_width,
            'y_min': y_min, 'y_max': y_max, 'y_height': y_height}

    return limits


def concatenate_dlc_data(dlc_data):
    for i, d in enumerate(dlc_data.keys()):       
        if i==0:
            dlc_data_concat = dlc_data[d]
        
        else:
            dlc_data_concat = pd.concat([dlc_data_concat, dlc_data[d]], 
                    ignore_index=True)            

    return dlc_data_concat

def concatenate_dlc_data_by_goal(dlc_data, behaviour_data):
    
    dlc_data_concat_by_goal = {}
    
    for g in behaviour_data.keys():
        
        for i, t in enumerate(behaviour_data[g]):
            if i==0:
                dlc_data_concat_by_goal[g] = dlc_data[t]
            else:
                dlc_data_concat_by_goal[g] = pd.concat([dlc_data_concat_by_goal[g], 
                                                dlc_data[t]], ignore_index=True) 
    return dlc_data_concat_by_goal


def get_directional_occupancy(directions, durations, n_bins=24):

    # create 24 bins, each 15 degrees
    direction_bins_og = np.linspace(-np.pi, np.pi, n_bins+1)
    direction_bins = direction_bins_og.copy()
    direction_bins[0] = direction_bins_og[0] - 0.1 # subtract a small number from the first bin so that the first value is included in the bin
    direction_bins[-1] = direction_bins_og[-1] + 0.1 # add a small number to the last bin so that the last value is included in the bin

    # get the bin indices for each value in directions
    bin_indices = np.digitize(directions, direction_bins, right=True) - 1
    # any bin_indices that are -1 should be 0
    bin_indices[bin_indices==-1] = 0
    # any bin_indices that are n_bins should be n_bins-1
    bin_indices[bin_indices==n_bins] = n_bins-1

    # get the occupancy for each bin, so a vector of length n_bins
    occupancy = np.zeros(n_bins)
    for i in range(n_bins):
        occupancy[i] = np.sum(durations[bin_indices==i])

    return occupancy, direction_bins_og


def get_directional_occupancy_from_dlc(dlc_data):

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
    # find columns that begin with 'relative_direction'
    for d in dlc_data.keys():
        if 'relative_direction' in d:
            direction_data['egocentric'].append(d)

    # directional_occupancy is a dictionary with keys 'allocentric' and 'egocentric'
    directional_occupancy = {'allocentric': {}, 'egocentric': {}}
    for direction_type in direction_data.keys():
        for d in direction_data[direction_type]:

            occupancy, direction_bins = get_directional_occupancy(dlc_data[d], dlc_data['durations'], n_bins=24)
            directional_occupancy[direction_type][d] = occupancy

    directional_occupancy = {'occupancy': directional_occupancy, 'bins': direction_bins} 

    return directional_occupancy


def plot_directional_occupancy(directional_occupancy, figure_dir):
    
    # if figure_dir doesn't exist, create it
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

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
            # concatenate the first value to the end so that the plot is closed
            occupancy = np.append(occupancy, occupancy[0])

            plt.polar(tick_positions, occupancy)

            plt.title(d)
            plt.show()
            
            # save the figure
            fig_path = os.path.join(figure_dir, f'{direction_type}_{d}.png')
            plt.savefig(fig_path)
            plt.close()


if __name__ == "__main__":
    animal = 'Rat46'
    session = '20-02-2024'
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

    # calculate frame intervals
    dlc_data = calculate_frame_durations(dlc_data)

    # concatenate dlc_data
    dlc_data_concat = concatenate_dlc_data(dlc_data)

    # get axes limits
    limits = get_axes_limits(dlc_data_concat)

    # calculate positional occupancy
    positional_occupancy = get_positional_occupancy(dlc_data_concat, limits)
    # save the positional_occupancy
    save_pickle(positional_occupancy, 'positional_occupancy', dlc_dir)
    # positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

    # plot the the trial paths
    for d in dlc_data.keys():
        plot_trial_path(dlc_data[d], limits, dlc_dir, d)

    # plot the heat map of occupancy
    plot_occupancy_heatmap(positional_occupancy, goal_coordinates, dlc_dir)
    
    # calculate directional occupancy
    directional_occupancy = get_directional_occupancy_from_dlc(dlc_data_concat)  
    # save the directional_occupancy
    save_pickle(directional_occupancy, 'directional_occupancy', dlc_dir)

    # plot the directional occupancy
    figure_dir = os.path.join(dlc_dir, 'directional_occupancy_plots')
    plot_directional_occupancy(directional_occupancy, figure_dir)

    # get directional occupancy by position
    directional_occupancy_by_position = get_directional_occupancy_by_position(dlc_data_concat, limits)
    # save the directional_occupancy_by_position
    save_pickle(directional_occupancy_by_position, 'directional_occupancy_by_position', dlc_dir)

    # calculate occupancy by goal
    behaviour_dir = get_behaviour_dir(data_dir)
    behaviour_data = load_pickle('behaviour_data_by_goal', behaviour_dir)
    dlc_data_concat_by_goal = concatenate_dlc_data_by_goal(dlc_data, behaviour_data)
    # save dlc_data_concat_by_goal
    save_pickle(dlc_data_concat_by_goal, 'dlc_data_concat_by_goal', dlc_dir)

    positional_occupancy_by_goal = {}
    directional_occupancy_by_goal = {}

    for g in behaviour_data.keys():
        
        # calculate positional occupancy
        positional_occupancy_by_goal[g] = \
            get_positional_occupancy(dlc_data_concat_by_goal[g], limits)
        
        figure_dir = os.path.join(dlc_dir, 'positional_occupancy_by_goal', f'goal_{g}')
        plot_occupancy_heatmap(positional_occupancy_by_goal[g], goal_coordinates, figure_dir)

        # calculate directional occupancy
        directional_occupancy_by_goal[g] = \
            get_directional_occupancy_from_dlc(dlc_data_concat_by_goal[g])  
        
        figure_dir = os.path.join(dlc_dir, 'directional_occupancy_by_goal', f'goal_{g}')        
        plot_directional_occupancy(directional_occupancy_by_goal[g], figure_dir)
   
    # save the positional_occupancy_by_goal
    save_pickle(positional_occupancy_by_goal, 'positional_occupancy_by_goal', dlc_dir)
    # save the directional_occupancy_by_goal
    save_pickle(directional_occupancy_by_goal, 'directional_occupancy_by_goal', dlc_dir)

    pass