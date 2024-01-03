import os
import glob
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats
from random import sample
import itertools
from get_directories import get_data_dir 

def classify_neurons():
    pass

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    
    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    spike_file  = os.path.join(spike_dir, 'unit_spike_times.pickle')
    with open(spike_file, 'rb') as handle:
        units = pickle.load(handle)

    # hardcoding the directory of the neuropixel bin file, as
    # it's not in the data directory
    bin_dir = '/media/jake/LaCie1/' + animal + '/' + session
    bin_file = glob.glob(bin_dir + '/*.ap.bin')[0]

    

    pass