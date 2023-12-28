import os
import glob
import numpy as np
import pandas as pd
import pickle
import re
from scipy import interpolate
import copy
from get_directories import get_data_dir
from process_dlc_data import load_dlc_processed_pickle
from get_pulses import load_bonsai_pulses





if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load the dlc processed data. It contains the video times in ms
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(dlc_pickle_path)
    n_trials = len(dlc_processed_data)    

    # load the pulses, which contains both the bonsai and spikeglx pulses in 
    # ms and samples, respectively
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    pulses = load_bonsai_pulses(bonsai_dir)

    # get the video times in samples
    video_samples = [None]*n_trials
    for i, d in enumerate(dlc_processed_data):
        video_samples[i] = d.columns[0][0]

    
    
    pass