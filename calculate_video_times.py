import os
import glob
import numpy as np
import pandas as pd
import pickle
import re
from scipy import interpolate
import copy
from get_directories import get_home_dir, get_data_dir





if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    dlc_dir = os.path.join(data_dir, 'deeplabcut')