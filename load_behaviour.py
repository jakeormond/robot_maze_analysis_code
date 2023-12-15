import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import pickle
import re 
from get_directories import get_data_dir 




if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')

    # find csv files in behaviour directory
    csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))

    # get the goals
    

    pass
