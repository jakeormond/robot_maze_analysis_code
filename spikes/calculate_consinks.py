import sys
import os
import numpy as np

import platform

# if on Windows
if platform.system() == 'Windows':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
# if on Linux
elif platform.system() == 'Linux':
    sys.path.append('/home/Jake/Documents/python_code/robot_maze_analysis_code')

from utilities.get_directories import get_data_dir, get_robot_maze_directory
from utilities.load_and_save_data import load_pickle, save_pickle
from behaviour.load_behaviour import split_dictionary_by_goal
from position.calculate_pos_and_dir import get_goal_coordinates, get_x_and_y_limits, cm_per_pixel
from position.calculate_occupancy import get_relative_direction_occupancy_by_position, concatenate_dlc_data, get_axes_limits, calculate_frame_durations
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials


if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)
    dlc_data = calculate_frame_durations(dlc_data)
    dlc_data_concat = concatenate_dlc_data(dlc_data)

    # get x and y limits
    limits = get_axes_limits(dlc_data_concat)

    # get relative direction occupancy by position if np array not already saved
    if os.path.exists(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy')) == False:
        reldir_occ_by_pos, sink_bins, candidate_sinks = get_relative_direction_occupancy_by_position(dlc_data_concat, limits)
        np.save(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy'), reldir_occ_by_pos)
        # save sink bins and candidate sinks as pickle files
        save_pickle(sink_bins, 'sink_bins', dlc_dir)
        save_pickle(candidate_sinks, 'candidate_sinks', dlc_dir)        

    else:
        reldir_occ_by_pos = np.load(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy'))
        sink_bins = load_pickle('sink_bins', dlc_dir)
        candidate_sinks = load_pickle('candidate_sinks', dlc_dir)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_by_goal', spike_dir)

    goals = units.keys()
    for goal in goals:
        goal_units = units[goal]
        
        for unit in goal_units.keys():
            unit = concatenate_unit_across_trials(unit)

            # get head directions
            head_directions = dlc_data_concat['hd']
            # make head_directions a numpy array
            head_directions = head_directions.to_numpy()
            
            



    pass