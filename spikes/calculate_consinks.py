import sys
import os

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


if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # get x and y limits
    x_and_y_limits = get_x_and_y_limits(dlc_data)