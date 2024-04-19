import os
import numpy as np
import pycircstat as circ
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import get_behaviour_dir
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, cm_per_pixel


cm_per_pixel = 0.2


def calculate_vector_fields(spike_rates_by_position_and_direction, x_bins, y_bins, direction_bins):

    bin_centres = direction_bins[:-1] + np.diff(direction_bins)/2

    vector_fields = {'units': {}, 'x_bins': x_bins, 
                     'y_bins': y_bins, 'direction_bins': direction_bins}
    mean_resultant_lengths = {'units': {}, 'x_bins': x_bins, 
                     'y_bins': y_bins, 'direction_bins': direction_bins} 
    

    # poss_unit_keys = ['units', 'popn']
    # data_keys = list(spike_rates_by_position_and_direction.keys())
    # # find the key common to both lists
    # unit_key = [k for k in poss_unit_keys if k in data_keys][0]

    units = list(spike_rates_by_position_and_direction.keys())

    for u in units:
        rates_by_pos_dir = spike_rates_by_position_and_direction[u]
        array_shape = rates_by_pos_dir.shape

        # initialize vector field as array of nans
        vector_field = np.full(array_shape[0:2], np.nan)
        mrl_field = np.full(array_shape[0:2], np.nan)

        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                rates = rates_by_pos_dir[i, j, :]
                
                # if any nan values in rates, OR if all values are zero skip
                if np.isnan(rates).any() or np.all(rates == 0):
                    continue

                mean_dir = np.round(circ.mean(bin_centres, rates), 3)
                mrl = np.round(circ.resultant_vector_length(bin_centres, rates), 3)

                if mean_dir > np.pi:
                    mean_dir = mean_dir - 2*np.pi

                vector_field[i, j] = mean_dir
                mrl_field[i, j] = mrl

        vector_fields['units'][u] = vector_field
        mean_resultant_lengths['units'][u] = mrl_field

    return vector_fields, mean_resultant_lengths


def calculate_vector_fields_2goals(spike_rates_by_position_and_direction_by_goal, behaviour_data):
    vector_fields_by_goal = {}
    mean_resultant_lengths_by_goal = {}
    for i, g in enumerate(behaviour_data.keys()):
        vector_fields_temp, mean_resultant_lengths_temp = calculate_vector_fields(spike_rates_by_position_and_direction_by_goal[g], 
                spike_rates_by_position_and_direction_by_goal['x_bins'], spike_rates_by_position_and_direction_by_goal['y_bins'], 
                spike_rates_by_position_and_direction_by_goal['direction_bins'])
        
        if i == 0:
            vector_fields_by_goal['x_bins'] = vector_fields_temp['x_bins']
            vector_fields_by_goal['y_bins'] = vector_fields_temp['y_bins']
            vector_fields_by_goal['direction_bins'] = vector_fields_temp['direction_bins']

            mean_resultant_lengths_by_goal['x_bins'] = mean_resultant_lengths_temp['x_bins']
            mean_resultant_lengths_by_goal['y_bins'] = mean_resultant_lengths_temp['y_bins']
            mean_resultant_lengths_by_goal['direction_bins'] = mean_resultant_lengths_temp['direction_bins']
        
        vector_fields_by_goal[g] = vector_fields_temp['units']
        mean_resultant_lengths_by_goal[g] = mean_resultant_lengths_temp['units']

    return vector_fields_by_goal, mean_resultant_lengths_by_goal


def plot_vector_field_test(vector_fields, plot_dir):
    x_bins = vector_fields['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title('test vector field')
    ax.set_xlabel('x position (cm)')
    ax.set_ylabel('y position (cm)')

    unit = list(vector_fields['units'].keys())[0]
    vector_field = vector_fields['units'][unit]

    # get number of elements in vector_field
    n_elements = vector_field.size

    # make a vector of the same size as vector_field with elements incrementing from -pi to pi
    increasing_directions = np.linspace(-np.pi, np.pi, n_elements)

    # arrange direction_bins into an array of the same shape as vector_field
    increasing_directions = increasing_directions.reshape(vector_field.shape)

    # plot vector field
    ax.quiver(x_centres, y_centres, np.cos(increasing_directions), 
              np.sin(increasing_directions), color='k', scale=15, 
              headlength=5, headaxislength=4, headwidth=4)

    # flip y axis
    ax.invert_yaxis()

    # increase the range of the axes by 10% to make room for the arrows
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    ax.set_xlim(x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range)
    ax.set_ylim(y_lim[0] - 0.1*y_range, y_lim[1] + 0.1*y_range)

    plt.show()

    fig.savefig(os.path.join(plot_dir, 'test vector field.png'))

    plt.close(fig)

    
def plot_vector_fields(vector_fields, plot_dir):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    x_bins = vector_fields['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2

    for u in vector_fields['units'].keys():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(u)
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel('y position (cm)')

        vector_field = vector_fields['units'][u]

        # plot vector field
        ax.quiver(x_centres, y_centres, np.cos(vector_field), np.sin(vector_field), color='k', scale=10)
        
        # show fig
        plt.show()
        
        # flip y axis
        ax.invert_yaxis()

        # increase the range of the axes by 10% to make room for the arrows
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        ax.set_xlim(x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range)
        ax.set_ylim(y_lim[0] - 0.1*y_range, y_lim[1] + 0.1*y_range)

        fig.savefig(os.path.join(plot_dir, f'{u}.png'))

        plt.close(fig)


def plot_vector_fields_2goals(vector_fields, goal_coordinates, x_centres, y_centres, plot_name, plot_dir, consink=None):

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle(plot_name, fontsize=24)

    goals = [g for g in vector_fields.keys() if isinstance(g, int)]
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
        
        
        vector_field = vector_fields[g][plot_name]

        # plot vector field
        ax[i].quiver(x_centres, y_centres, np.cos(vector_field), np.sin(vector_field), color='k', scale=10)
       
        # flip y axis
        ax[i].invert_yaxis()

        # increase the range of the axes by 10% to make room for the arrows
        x_lim = ax[i].get_xlim()
        y_lim = ax[i].get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        ax[i].set_xlim(x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range)
        ax[i].set_ylim(y_lim[0] - 0.1*y_range, y_lim[1] + 0.1*y_range)

        #

        
        if consink is not None:
            mrl = consink[g]['mrl']
            ci_95 = consink[g]['ci_95']
            ci_999  = consink[g]['ci_999']
            consink_pos = consink[g]['position']
            consink_angle = consink[g]['mean_angle']
            if consink_angle > np.pi:
                consink_angle = consink_angle - 2*np.pi
            
            # plot a filled circle at the consink position
            if mrl > ci_95:
                consink_color = 'r'
            else: # color is gray
                consink_color = 'gray'

            circle = plt.Circle((consink_pos[0], 
                consink_pos[1]), 50, color=consink_color, 
                fill=True)
            ax[i].add_artist(circle)      
        
            # add text with mrl, ci_95, ci_999
            # ax[i].text(0, 2100, f'mrl: {mrl:.2f}\nci_95: {ci_95:.2f}\nci_999: {ci_999:.2f}\nangle: {consink_angle:.2f}', fontsize=16)
            ax[i].text(0, 2100, f'mrl: {mrl:.2f}\nci_999: {ci_999:.2f}\nangle: {consink_angle:.2f}', fontsize=16)
            
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



        # set the axes to have identical scales
        ax[i].set_aspect('equal')        

    fig.savefig(os.path.join(plot_dir, f'{plot_name}.png'))


def plot_vector_fields_2goals_all_units(vector_fields, goal_coordinates, plot_dir, consinks=None):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # goals are the two numeric keys in vector_fields
    goals = [g for g in vector_fields.keys() if isinstance(g, int)]
    
    x_bins = vector_fields['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2

    units = list(vector_fields[goals[0]].keys())
    for u in units:
        if consinks is not None:
            consink = {goals[0]: None, goals[1]: None}
            # find row with unit index
            for g in goals:
                consink[g] = consinks[g].loc[u]
            for g in goals:
                mrl = consink[g]['mrl']
                ci_95 = consink[g]['ci_95']

                if mrl > ci_95 :
                    plot_vector_fields_2goals(vector_fields, goal_coordinates, x_centres, y_centres, u, plot_dir, consink=consink)
                    break
            
        else:
            # plot_vector_fields_2goals(vector_fields, goal_coordinates, x_centres, y_centres, u, plot_dir, consink=None)
            pass



if __name__ == "__main__":
    animal = 'Rat46'
    session = '20-02-2024'
    data_dir = get_data_dir(animal, session)

    # get goal coordinates
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')

    code_to_run = [2]

    ###################### VECTOR FIELDS ACROSS WHOLE SESSSION (I.E. NOT SPLIT BY GOAL) #####################
    if 1 in code_to_run:
        # load spike_rates_by_position_and_direction
        spike_rates_by_position_and_direction = load_pickle('spike_rates_by_position_and_direction', spike_dir)
        
        vector_fields, mean_resultant_lengths = calculate_vector_fields(spike_rates_by_position_and_direction)
        save_pickle(vector_fields, 'vector_fields', spike_dir)
        save_pickle(mean_resultant_lengths, 'mean_resultant_lengths', spike_dir)

        # load vector fields
        vector_fields = load_pickle('vector_fields', spike_dir)

        plot_dir = os.path.join(spike_dir, 'vector_fields')

        # plot test field
        plot_vector_field_test(vector_fields, plot_dir)

        # plot vector fields
        # plot_vector_fields(vector_fields, plot_dir)


    ########################## BOTH GOALS ############################
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    behaviour_data = load_pickle('behaviour_data_by_goal', behaviour_dir)

    if 2 in code_to_run:

        consinks = load_pickle('consinks_df', spike_dir)

        spike_rates_by_position_and_direction_by_goal = load_pickle('spike_rates_by_position_and_direction_by_goal', spike_dir)

        vector_fields_by_goal, mean_resultant_lengths_by_goal = calculate_vector_fields_2goals(spike_rates_by_position_and_direction_by_goal, behaviour_data)
            
        save_pickle(vector_fields_by_goal, 'vector_fields_by_goal', spike_dir)
        save_pickle(mean_resultant_lengths_by_goal, 'mean_resultant_lengths_by_goal', spike_dir)

        # plot vector fields by goal
        # plot_dir = os.path.join(spike_dir, 'vector_fields_by_goal')
        plot_dir = os.path.join(spike_dir, 'vector_fields_by_goal', 'consinks')

        plot_vector_fields_2goals_all_units(vector_fields_by_goal, goal_coordinates, plot_dir, consinks=consinks)


    ############################# BOTH GOALS - COMBINED PRINCIPAL CELLS ############################
    
    if 3 in code_to_run:

        # load data 
        spike_rates_by_position_and_direction_by_goal_popn = load_pickle('spike_rates_by_position_and_direction_by_goal_popn', spike_dir)

        vector_fields_by_goal, mean_resultant_lengths_by_goal = calculate_vector_fields_2goals(spike_rates_by_position_and_direction_by_goal_popn, behaviour_data)
        
        save_pickle(vector_fields_by_goal, 'vector_fields_by_goal_popn', spike_dir)
        save_pickle(mean_resultant_lengths_by_goal, 'mean_resultant_lengths_by_goal_popn', spike_dir)

        # plot vector fields by goal
        plot_dir = os.path.join(spike_dir, 'vector_fields_by_goal')
        plot_name = 'pyramidal_popn'
        plot_vector_fields_2goals_all_units(vector_fields_by_goal, goal_coordinates, plot_dir)

    pass

        











    