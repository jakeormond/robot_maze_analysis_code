import os
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir
from utilities.load_and_save_data import load_pickle, save_pickle
from position.calculate_pos_and_dir import get_goal_coordinates

cm_per_pixel = 0.2

def place_field_centre_of_mass(rate_map, x_bins, y_bins):
    """
    Calculate the centre of mass of a rate map.
    
    Parameters
    ----------
    rate_map : 2D numpy array
        The rate map.
    x_bins : 1D numpy array
        The x bin edges.
    y_bins : 1D numpy array
        The y bin edges.
        
    Returns
    -------
    com_x : float
        The x coordinate of the centre of mass.
    com_y : float
        The y coordinate of the centre of mass.
    """
    
    # convert all nans in rate_map to 0s
    rate_map = rate_map.copy()
    rate_map[np.isnan(rate_map)] = 0
   
    # calculate the centre of mass
    com_x = 0
    com_y = 0
    total_rate = 0
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            com_x += rate_map[i, j] * (x_bins[i] + x_bins[i + 1]) / 2
            com_y += rate_map[i, j] * (y_bins[j] + y_bins[j + 1]) / 2
            total_rate += rate_map[i, j]
    com_x /= total_rate
    com_y /= total_rate

    return com_x, com_y

def place_field_distance_to_goal(rate_map, x_bins, y_bins, goal_coordinates=None):
    """
    Calculate the distance from the centre of mass of a rate map to the goal.
    
    Parameters
    ----------
    rate_map : 2D numpy array
        The rate map.
    x_bins : 1D numpy array
        The x bin edges.
    y_bins : 1D numpy array
        The y bin edges.
    goal_coordinates : dict
        The goal coordinates.
        
    Returns
    -------
    distance : float
        The distance from the centre of mass to the goal
    """  

    if goal_coordinates is None:
        raise ValueError('goal_coordinates must be provided')

    com_x, com_y = place_field_centre_of_mass(rate_map, x_bins, y_bins)
    
    goals = list(goal_coordinates.keys())
    distance_to_goal = {}
   
    for g in goals:
        
        goal_coordinates_ = goal_coordinates[g]
        goal_x, goal_y = goal_coordinates_

        distance_to_goal[g] = ((com_x - goal_x)**2 + (com_y - goal_y)**2)**0.5

    return distance_to_goal

def calculate_rate_map_property_across_goals(func, rate_maps, **kwargs):
    goals = list(rate_maps.keys())
    properties = {}

    for goal in goals:
        properties[goal] = {}
        x_bins = rate_maps[goal]['x_bins']
        y_bins = rate_maps[goal]['y_bins']
        units = list(rate_maps[goal]['rate_maps'].keys())

        for unit in units:
            rate_map = rate_maps[goal]['rate_maps'][unit]
            properties[goal][unit] = func(rate_map, x_bins, y_bins, **kwargs)

    return properties


def plot_place_field_positions():

    pass

def place_field_distances_to_goal_swarmplot(distances_to_goal):
    """
        For each of the two goal epochs, plot the place field distance to each goal,
        producing four swarmplots in total.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    # get the goal names
    goals = list(distances_to_goal.keys())

    # get the unit names
    units = list(distances_to_goal[goals[0]].keys())

    # create a figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, goal in enumerate(goals):
        for j, unit in enumerate(units):
            distances = [distances_to_goal[goal][unit][g] for g in goals]
            sns.swarmplot(data=distances, ax=axs[i, j])
            axs[i, j].set_title(f'Goal: {goal}, Unit: {unit}')
            axs[i, j].set_ylabel('Distance to goal (cm)')

    plt.tight_layout()
    plt.show()

    return fig




def main():
    animal = 'Rat47'
    session = '08-02-2024'
    data_dir = get_data_dir(animal, session)

    # load rate maps
    rate_maps = load_pickle('smoothed_rate_maps_by_goal', os.path.join(data_dir, 'spike_sorting'))

    # calculate the centre of mass for each rate map
    coms = calculate_rate_map_property_across_goals(place_field_centre_of_mass, rate_maps)
    
    # get the 2 goal positions
    goals = list(rate_maps.keys())

    # get goal coordinates
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)

    # get the distance to each goal for each place field
    distances_to_goal = calculate_rate_map_property_across_goals(place_field_distance_to_goal, rate_maps, goal_coordinates=goal_coordinates)

    # load neuron types
    neuron_types = load_pickle('neuron_types', os.path.join(data_dir, 'spike_sorting'))

    # restrict distances_to_goal to pyramidal cells
    distances_to_goal = {g: {u: distances_to_goal[g][u] for u in distances_to_goal[g] if u in neuron_types and neuron_types[u] == 'pyramidal'} for g in distances_to_goal}

    # plot the place field distances to goal
    place_field_distances_to_goal_swarmplot(distances_to_goal)

    pass


if __name__ == '__main__':
    main()