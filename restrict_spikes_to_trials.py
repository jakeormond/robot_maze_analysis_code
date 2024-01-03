import os
import numpy as np
import pandas as pd
import pickle

from get_directories import get_data_dir 
from process_dlc_data import load_dlc_processed_pickle

def restrict_spikes_to_trials():
    pass


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(dlc_pickle_path)

    video_dir = os.path.join(data_dir, 'video_files')
    video_endpoints_path = os.path.join(video_dir, 'video_endpoints.pkl')
    with open(video_endpoints_path, 'rb') as f:
        video_endpoints = pickle.load(f)

    # load the spike data
    unit_dir = os.path.join(data_dir, 'spike_sorting')
    
  