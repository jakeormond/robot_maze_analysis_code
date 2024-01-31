# code for generating the dataset for use with PyTorch
import os 
import numpy as np

from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from load_behaviour import get_behaviour_dir

sample_freq = 30000 # Hz


def smooth_positional_data(dlc_data, window_size=100):  
    # THIS IS JUST A PLACEHOLDER FOR NOW   
    pass


def create_positional_trains(dlc_data, window_size=100): # we'll default to 100 ms windows for now
    # first, get the start and end time of the video
    window_in_samples = window_size * sample_freq / 1000 # convert window size to samples
    for k in dlc_data.keys():
        start_time = dlc_data[k].video_samples.iloc[0]
        end_time = dlc_data[k].video_samples.iloc[-1]
        duration = end_time - start_time

        # calculate the number of windows. Windows overlap by 50%.
        num_windows = int(np.ceil(duration/window_in_samples)) * 2 - 1

        # get the bin edges for the windows
        window_edges = [start_time + i*window_in_samples/2 for i in range(num_windows+1)]


        pass



    # we'll use a moving average over about 3 video frames to smooth the positional data

    pass









if __name__ == "__main__":
    animal = 'Rat65'
    session = '10-11-2023'
    data_dir = get_data_dir(animal, session)
    
    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_w_behav_correlates', spike_dir)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # create positional and spike trains with overlapping windows
    # and save as a pickle file
    windowed_dlc = create_positional_trains(dlc_data, window_size=100)


    pass 