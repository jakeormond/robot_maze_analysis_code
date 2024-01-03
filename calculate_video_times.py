import os
import glob
import numpy as np
import pandas as pd
import pickle
import re
import copy
from get_directories import get_data_dir
from process_dlc_data import load_dlc_processed_pickle
from get_pulses import load_bonsai_pulses

# note that sample rate is hardcoded as 30000
# create global variable for sample rate
sample_rate = 30000

def get_video_times_in_samples(dlc_processed_data, pulses):
    """
    Get the video times in samples. This is done by interpolating the video 
    timestamps with the bonsai timestamps. The last few frames after the last 
    pulse can't be interpolated properly so they need to be manually calculated.
    
    Parameters
    ----------
    dlc_processed_data : list of pandas dataframes
        The dlc processed data. Each element in the list is a trial. Each 
        dataframe contains the dlc processed data for a trial. Contains only
        video times in ms, not in samples, which are necessary to calculate
        the times of the recorded spikes. 
    pulses : list of pandas dataframes
        The pulse times and samples. Each element in the list is a trial. 
        
    Returns
    -------
    dlc_processed_with_samples : list of pandas dataframes
        The dlc processed data with the video times in samples. Each element in 
        the list is a trial. Each dataframe contains the dlc processed data for 
        a trial. The first column is the video time in ms and the second column 
        is the video time in samples. The rest of the columns are the body 
        part positions and head direction.
    """
    n_trials = len(dlc_processed_data)
    dlc_processed_with_samples = [None]*n_trials

    for i, d in enumerate(dlc_processed_data):
        video_time = d.columns[0][0]
        pulse_time = pulses[i].name

        if video_time != pulse_time:
            # raise exception
            raise ValueError(
                f"Video time and pulse time do not match. "
                f"Video time: {video_time}, Pulse time: {pulse_time}"
            )
        
        video_ts = d[video_time, 'ts']
        pulses_ts = pulses[i]['bonsai_pulses_ms']
        pulses_samples = pulses[i]['imec_pulses_samples']

        # interpolate the video samples 
        video_samples = np.interp(video_ts, pulses_ts, pulses_samples)
        video_samples = np.round(video_samples).astype(int)

        # the last few frames after the last pulse can't be interpolated properly
        # so they need to be manually calculated
        # first find the first video_ts that is greater than the last pulse_ts
        last_pulse_ts = pulses_ts.iloc[-1]
        last_video_ind = np.where(video_ts > last_pulse_ts)[0][0]
        video_samples[last_video_ind:] = video_samples[last_video_ind - 1] + \
            (video_ts[last_video_ind:] - video_ts[last_video_ind - 1])*(sample_rate/1000) # sample rate 30000 divide by 1000 (i.e. 30 samples per ms)

        # add the video samples to the dlc_processed_data.
        # make it the second column
        dlc_processed_with_samples[i] = copy.deepcopy(d)
        dlc_processed_with_samples[i].insert(1, (video_time, 'samples'), 
                                             video_samples)
        
    return dlc_processed_with_samples



if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load the dlc processed data. It contains the video times in ms
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(dlc_pickle_path)

    # load the pulses, which contains both the bonsai and spikeglx pulses in 
    # ms and samples, respectively
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    pulses = load_bonsai_pulses(bonsai_dir)

    dlc_processed_with_samples = get_video_times_in_samples(dlc_processed_data, pulses)
       
    # save the processed data to a pickle file
    pickle_path = os.path.join(dlc_dir, 'dlc_processed_data_with_samples.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(dlc_processed_with_samples, f)

    # delete dlc_processed_with samples
    del dlc_processed_with_samples

    # load the processed data with samples
    with open(pickle_path, 'rb') as f:
        dlc_processed_with_samples = pickle.load(f)
    
    pass