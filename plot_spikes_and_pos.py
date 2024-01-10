import os
import numpy as np
import pandas as pd
import pickle

from get_directories import get_data_dir, get_robot_maze_directory



if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    restricted_units_file = os.path.join(unit_dir, 'restricted_units.pickle')
    with open(restricted_units_file, 'rb') as handle:
        restricted_units = pickle.load(handle)