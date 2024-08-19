import os
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir, get_robot_maze_directory
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import split_dictionary_by_goal
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits

cm_per_pixel = 0.2


def basic_spike_pos_plot(ax, unit, dlc_data, goal_coordinates, x_and_y_limits):
    
    # plot the goal positions
    colours = ['b', 'g']

    # if goal_coordinates is a dictionary, then there is only one goal
    if isinstance(goal_coordinates, dict):
        for i, g in enumerate(goal_coordinates.keys()):
            # draw a circle with radius 80 around the goal on ax
            circle = plt.Circle((goal_coordinates[g][0], 
                goal_coordinates[g][1]), 80, color=colours[i], 
                fill=False, linewidth=10)
            ax.add_artist(circle)       

        else:
            circle = plt.Circle((goal_coordinates[0], 
                goal_coordinates[1]), 80, color=colours[0], 
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
        # plt.close(fig)

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
        # plt.close(fig)


def plot_rate_maps(rate_maps, smoothed_rate_maps, goal_coordinates, plot_dir):
    # plot the rate maps
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    x_bins = rate_maps['x_bins']
    y_bins = rate_maps['y_bins']

    occupancy = rate_maps['occupancy']

    for u in rate_maps['rate_maps'].keys():
        rate_map = rate_maps['rate_maps'][u]
        rate_map_smoothed = smoothed_rate_maps['rate_maps'][u]        
        
        # make figure with 2 subplots
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))        

        # plot the unsmoothed rate map in the first subplot
        for i in range(2):
            if i == 0:
                im = ax[0].imshow(rate_map, cmap='jet', aspect='auto')
                ax[0].set_title([u + ' - unsmoothed'], fontsize=15)  
                # add a colourbar
                divider = make_axes_locatable(ax[0])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                                
            else:
                im = ax[1].imshow(rate_map_smoothed, cmap='jet', aspect='auto')
                ax[1].set_title([u + ' - smoothed'], fontsize=15)
                # add a colourbar
                divider = make_axes_locatable(ax[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(im, cax=cax)
                
                
                # cbar = plt.colorbar(ax[1].imshow(rate_map_smoothed, cmap='jet', aspect='auto'), ax=ax[1])

            cbar.set_label('Firing rate (Hz)', size=15)         
            cbar.ax.tick_params(labelsize=15)   

            ax[i].set_xlabel('x (cm)',fontsize=15)
            ax[i].set_ylabel('y (cm)', fontsize=15)
            # don't need to flip the y axis because it's an image, so plots from top dow
            ax[i].set_aspect('equal', 'box')

            # get x_ticks
            xticks = ax[i].get_xticks()
            # set the x ticks so that only those that are between 0 and n_x_bins are shown
            xticks = xticks[(xticks >= 0) & (xticks < len(x_bins))]
            ax[i].set_xticks(xticks)
            # interpolate the x values to get the pixel values, noting that 0.5 needs to be added to the xticks, because they are centred on their bins
            xtick_values = np.int32(np.round(np.interp(xticks + 0.5, np.arange(len(x_bins)), x_bins), 0))
            # then convert to cm 
            xtick_values = np.int32(np.round(xtick_values * cm_per_pixel, 0))        
            ax[i].set_xticklabels(xtick_values)
            ax[i].tick_params(axis='x', labelsize=15)

            # do the same for y_ticks
            yticks = ax[i].get_yticks()
            yticks = yticks[(yticks >= 0) & (yticks < len(y_bins))]
            ax[i].set_yticks(yticks)
            ytick_values = np.int32(np.round(np.interp(yticks + 0.5, np.arange(len(y_bins)), y_bins), 0))
            ytick_values = np.int32(np.round(ytick_values * cm_per_pixel, 0))
            ax[i].set_yticklabels(ytick_values)
            ax[i].tick_params(axis='y', labelsize=15)

            # draw the goal positions over top
            colours = ['k', '0.5']

            if isinstance(goal_coordinates, dict):
                for j, g in enumerate(goal_coordinates.keys()):
                    # first, convert to heat map coordinates
                    goal_x, goal_y = goal_coordinates[g]

                    # Convert to heat map coordinates
                    goal_x_heatmap = np.interp(goal_x, x_bins, np.arange(len(x_bins))) - 0.5
                    goal_y_heatmap = np.interp(goal_y, y_bins, np.arange(len(y_bins))) - 0.5                          
                    
                    # draw a circle with radius 80 around the goal on ax
                    circle = plt.Circle((goal_x_heatmap, 
                        goal_y_heatmap), radius=1, color=colours[j], 
                        fill=False, linewidth=4)
                    ax[i].add_artist(circle)
            else:
                goal_x, goal_y = goal_coordinates

                # Convert to heat map coordinates
                goal_x_heatmap = np.interp(goal_x, x_bins, np.arange(len(x_bins))) - 0.5
                goal_y_heatmap = np.interp(goal_y, y_bins, np.arange(len(y_bins))) - 0.5                          
                
                # draw a circle with radius 80 around the goal on ax
                circle = plt.Circle((goal_x_heatmap, 
                    goal_y_heatmap), radius=1, color=colours[0], 
                    fill=False, linewidth=4)
                ax[i].add_artist(circle)


        # show the plot
        # plt.show()
        
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))

        # plt.close()


def plot_rate_maps_2goals(rate_maps_by_goal, goal_coordinates, plot_dir):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    goals = list(rate_maps_by_goal.keys())
    # get list of units
    units = list(rate_maps_by_goal[goals[0]]['rate_maps'].keys())

    x_bins = rate_maps_by_goal[goals[0]]['x_bins'] # bins are the same for both goals
    y_bins = rate_maps_by_goal[goals[0]]['y_bins']

    for u in units:

        # get the max firing rate across both goals, exluding nans
        max_rate_goal1 = np.nanmax(rate_maps_by_goal[goals[0]]['rate_maps'][u])
        max_rate_goal2 = np.nanmax(rate_maps_by_goal[goals[1]]['rate_maps'][u])     
       
        max_rate = np.around(np.max([max_rate_goal1, max_rate_goal2]), 1)
                
        # make figure with 2 subplots
        fig, ax = plt.subplots(1, 2, figsize=(20, 10)) 

        # plot the unsmoothed rate map in the first subplot
        for i in range(2):
            rate_map = rate_maps_by_goal[goals[i]]['rate_maps'][u]
            
            im = ax[i].imshow(rate_map, cmap='jet', aspect='auto')
            
            # Set the color limits of the heatmap
            im.set_clim(0.0, max_rate)

            ax[i].set_title([u + f' - goal_{goals[i]}'], fontsize=15)  
            # add a colourbar
            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
                                
            cbar.set_label('Firing rate (Hz)', size=15)                
            cbar.ax.tick_params(labelsize=15)   

            ax[i].set_xlabel('x (cm)',fontsize=15)
            ax[i].set_ylabel('y (cm)', fontsize=15)
            # don't need to flip the y axis because it's an image, so plots from top dow
            ax[i].set_aspect('equal', 'box')

            # get x_ticks
            xticks = ax[i].get_xticks()
            # set the x ticks so that only those that are between 0 and n_x_bins are shown
            xticks = xticks[(xticks >= 0) & (xticks < len(x_bins))]
            ax[i].set_xticks(xticks)
            # interpolate the x values to get the pixel values, noting that 0.5 needs to be added to the xticks, because they are centred on their bins
            xtick_values = np.int32(np.round(np.interp(xticks + 0.5, np.arange(len(x_bins)), x_bins), 0))
            # then convert to cm 
            xtick_values = np.int32(np.round(xtick_values * cm_per_pixel, 0))        
            ax[i].set_xticklabels(xtick_values)
            ax[i].tick_params(axis='x', labelsize=15)

            # do the same for y_ticks
            yticks = ax[i].get_yticks()
            yticks = yticks[(yticks >= 0) & (yticks < len(y_bins))]
            ax[i].set_yticks(yticks)
            ytick_values = np.int32(np.round(np.interp(yticks + 0.5, np.arange(len(y_bins)), y_bins), 0))
            ytick_values = np.int32(np.round(ytick_values * cm_per_pixel, 0))
            ax[i].set_yticklabels(ytick_values)
            ax[i].tick_params(axis='y', labelsize=15)

            # draw the goal positions over top
            colours = ['k', '0.5']
            for j, g in enumerate(goal_coordinates.keys()):
                # first, convert to heat map coordinates
                goal_x, goal_y = goal_coordinates[g]

                # Convert to heat map coordinates
                goal_x_heatmap = np.interp(goal_x, x_bins, np.arange(len(x_bins))) - 0.5
                goal_y_heatmap = np.interp(goal_y, y_bins, np.arange(len(y_bins))) - 0.5                          
                
                # draw a circle with radius 80 around the goal on ax
                circle = plt.Circle((goal_x_heatmap, 
                    goal_y_heatmap), radius=1, color=colours[j], 
                    fill=False, linewidth=4)
                ax[i].add_artist(circle)


        # show the plot
        # plt.show()
        
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        # plt.close()

    pass


def plot_spike_rates_by_direction(spike_rates_by_direction, plot_dir):

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # note that polar plot converts radian to degrees. 0 degrees = 0 radians,
    # 90 degrees = pi/2 radians, 180 degrees = +/- pi radians, etc.
        
    bins = spike_rates_by_direction['bins']
    # polar plot ticks are the centres of the bins
    tick_positions = np.round(bins[:-1] + np.diff(bins)/2, 2)
    tick_positions = np.append(tick_positions, tick_positions[0])

    spike_rates = spike_rates_by_direction['units']

    for u in spike_rates.keys():
        # create plot with 8 subplots. Currently we only 
        # need 7 plots, but easiest to arrange in two rows of 4

        fig, ax = plt.subplots(2, 4, figsize=(20, 10), subplot_kw=dict(polar=True))
        
        for i, d in enumerate(spike_rates[u].keys()):

            # get the spike rates for this direction
            spike_rates_temp = spike_rates[u][d]
            # concatenate the first value to the end so that the plot is closed
            spike_rates_temp = np.append(spike_rates_temp, spike_rates_temp[0])

            # make a polar plot
            ax[i//4, i%4].plot(tick_positions, spike_rates_temp, 'b-')
            # ax[i//4, i%4].polar(tick_positions, spike_rates_temp, 'k.-', markersize=10)
            # ax[i//4, i%4].set_xticks(tick_positions)
            # ax[i//4, i%4].set_xticklabels(tick_positions)
            # ax[i//4, i%4].set_ylim([0, 10])
            # ax[i//4, i%4].set_yticks([0, 5, 10])
            ax[i//4, i%4].tick_params(axis='x', labelsize=15) # these are the degrees
            ax[i//4, i%4].tick_params(axis='y', labelsize=15) # these are the rates
            
            if d != 'hd':
                ax[i//4, i%4].set_theta_zero_location('N')
            # ax[i//4, i%4].set_theta_direction(-1)

            ax[i//4, i%4].set_title(f'{u} - {d}', fontsize=15)

        # plt.show()
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        # plt.close()

    pass


def plot_spike_rates_by_direction_2goals(spike_rates_by_direction, plot_dir):

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    goals = list(spike_rates_by_direction.keys())

    # note that polar plot converts radian to degrees. 0 degrees = 0 radians,
    # 90 degrees = pi/2 radians, 180 degrees = +/- pi radians, etc.
        
    bins = spike_rates_by_direction[goals[0]]['bins']
    # polar plot ticks are the centres of the bins
    tick_positions = np.round(bins[:-1] + np.diff(bins)/2, 2)
    tick_positions = np.append(tick_positions, tick_positions[0])

    units = list(spike_rates_by_direction[goals[0]]['units'].keys())

    for u in units:
        # create plot with 2 rows of 7 polar plots
        fig, ax = plt.subplots(2, 7, figsize=(20, 10), subplot_kw=dict(polar=True))
        plt.subplots_adjust(wspace=0.5)

        for i, g in enumerate(goals):
            spike_rates = spike_rates_by_direction[g]['units'][u]
        
            for j, d in enumerate(spike_rates.keys()):

                # get the spike rates for this direction
                spike_rates_temp = spike_rates[d]
                # concatenate the first value to the end so that the plot is closed
                spike_rates_temp = np.append(spike_rates_temp, spike_rates_temp[0])

                # make a polar plot
                ax[i, j].plot(tick_positions, spike_rates_temp, 'b-')
                # ax[i//4, i%4].polar(tick_positions, spike_rates_temp, 'k.-', markersize=10)
                # ax[i//4, i%4].set_xticks(tick_positions)
                # ax[i//4, i%4].set_xticklabels(tick_positions)
                # ax[i//4, i%4].set_ylim([0, 10])
                # ax[i//4, i%4].set_yticks([0, 5, 10])
                ax[i, j].tick_params(axis='x', labelsize=8) # these are the degrees
                ax[i, j].tick_params(axis='y', labelsize=8) # these are the rates
                
                if d != 'hd':
                    ax[i, j].set_theta_zero_location('N')
                # ax[i//4, i%4].set_theta_direction(-1)

                ax[i, j].set_title(f'{u} - {d}', fontsize=8)

        # plt.show()
        fig.savefig(os.path.join(plot_dir, f'{u}.png'))
        # plt.close()

    pass
    
    
if __name__ == "__main__":
    
    experiment = 'robot_single_goal'
    animal = 'Rat_HC1'
    session = '31-07-2024'

    data_dir = get_data_dir(experiment, animal, session)

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

    
    # plot spikes and position 
    plot_dir = os.path.join(spike_dir, 'spikes_and_pos')

    # plot_spikes_and_pos(units, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)


    # plot spike rates by direction
    plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction')
    spike_rates_by_direction = load_pickle('spike_rates_by_direction', spike_dir)

    # plot_spike_rates_by_direction(spike_rates_by_direction, plot_dir)

    # plot rate maps
    plot_dir = os.path.join(spike_dir, 'rate_maps')
    rate_maps = load_pickle('rate_maps', spike_dir)
    smoothed_rate_maps = load_pickle('smoothed_rate_maps', spike_dir)  
    plot_rate_maps(rate_maps, smoothed_rate_maps, goal_coordinates, plot_dir)

    
    
    
    
    
    
    # plot spike and position by goal
    units_by_goal = {}
    for u in units.keys():
        units_by_goal[u] = split_dictionary_by_goal(units[u], data_dir)
    
    plot_dir = os.path.join(spike_dir, 'spikes_and_pos_by_goal')
    plot_spikes_2goals(units_by_goal, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)

    # plot spike rates by direction
    plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction')
    spike_rates_by_direction = load_pickle('spike_rates_by_direction', spike_dir)

    plot_spike_rates_by_direction(spike_rates_by_direction, plot_dir)

    # plot rate maps 
    plot_dir = os.path.join(spike_dir, 'rate_maps')
    rate_maps = load_pickle('rate_maps', spike_dir)
    smoothed_rate_maps = load_pickle('smoothed_rate_maps', spike_dir)  
    plot_rate_maps(rate_maps, smoothed_rate_maps, goal_coordinates, plot_dir)

    # plot smoothed rate maps by goal
    plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal')
    rate_maps_by_goal = load_pickle('smoothed_rate_maps_by_goal', spike_dir)
    plot_rate_maps_2goals(rate_maps_by_goal, goal_coordinates, plot_dir)

    # plot spike rates by direction by goal
    spike_rates_by_direction = load_pickle('spike_rates_by_direction_by_goal', spike_dir)
    plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction_by_goal')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    plot_spike_rates_by_direction_2goals(spike_rates_by_direction, plot_dir)

    for g in spike_rates_by_direction.keys():
        plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction_by_goal', f'goal_{g}')
        plot_spike_rates_by_direction(spike_rates_by_direction[g], plot_dir)

    # plot rate maps by goal
    rate_maps_by_goal = load_pickle('rate_maps_by_goal', spike_dir)
    smoothed_rate_maps_by_goal = load_pickle('smoothed_rate_maps_by_goal', spike_dir)

    plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal')
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    for g  in rate_maps_by_goal.keys():
        plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal', f'goal_{g}')     
        plot_rate_maps(rate_maps_by_goal[g], smoothed_rate_maps_by_goal[g], goal_coordinates, plot_dir)

    
    # create combined plots, with goal 1 in subplot 1 and goal 2 in subplot 2
    plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal', 'both_goals') 
    plot_rate_maps_2goals(rate_maps_by_goal, goal_coordinates, plot_dir)    

    

        
    pass