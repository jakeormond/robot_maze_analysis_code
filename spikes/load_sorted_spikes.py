import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir 
from utilities.load_and_save_data import save_pickle

sample_freq = 30000

''' 
spike sorting files required:

- cluster_group.tsv
- spike_clusters.npy
- spike_times.npy

'''

def load_sorted_spikes(spike_dir):
    cluster_quality = pd.read_csv(os.path.join(spike_dir, 
                            'cluster_group.tsv'), sep='\t')
    # spike_templates = np.load(os.path.join(spike_dir, 'spike_templates.npy'))
    # templates = np.load(os.path.join(spike_dir, 'templates.npy'))
    clusters = np.load(os.path.join(spike_dir, 'spike_clusters.npy'))
    spike_times = np.load(os.path.join(spike_dir, 'spike_times.npy'))

    good_clusters = cluster_quality['cluster_id'].loc[cluster_quality['group'] == 'good']
    good_clusters = good_clusters.reset_index(drop=True)

    units = {}

    for indx in range(len(good_clusters)):
        cluster_id = good_clusters[indx]
    
        spike_ind = np.nonzero(clusters == cluster_id)[0]

        spike_samples_temp = np.squeeze(spike_times[spike_ind])
    
        # convert to seconds - RIGHT NOW, WE WON'T; WE CAN DO IT ON THE FLY
        # spike_times_temp = np.around(spike_samples_temp/sample_freq, 3)
        
        # data_for_df = {'samples': spike_samples_temp, 'times': spike_times_temp}
        data_for_df = {'samples': spike_samples_temp}
        unit_df = pd.DataFrame(data_for_df)
               
        units[f'cluster_{cluster_id}'] = unit_df      

    return units


def correct_spike_times(units, n_samples_and_pulses, bin_file_df):

    corrected_units = {}

    for u in units.keys():
        unit = units[u]
        unit_samples = unit['samples'].values

        corrected_samples = []

        # loop through the bin files and correct the spike times
        for i in range(len(bin_file_df)):
            if i == 0:
                start_sample = 0

            else:
                start_sample = bin_file_df['cumulative_samples'].iloc[i-1]
            
            end_sample = bin_file_df['cumulative_samples'].iloc[i] - 1

            # get unit_samples that are in the bin file
            unit_samples_in_bin = unit_samples[(unit_samples >= start_sample) & (unit_samples <= end_sample)]

            # if there are no samples in the bin file, skip to the next bin file
            if len(unit_samples_in_bin) == 0:
                continue

            # convert there samples to non-cumulative samples
            if i != 0:
                unit_samples_in_bin = unit_samples_in_bin - bin_file_df['cumulative_samples'].iloc[i-1]

            # find the index of the bin_file in n_samples_and_pulses
            bin_file = bin_file_df['bin_file'].iloc[i]
            bin_file_index = n_samples_and_pulses.index[n_samples_and_pulses['dir_name'] == bin_file].tolist()[0]

            if bin_file_index != 0:
                unit_samples_in_bin = unit_samples_in_bin + n_samples_and_pulses['cumulative_samples'].iloc[bin_file_index-1]

            corrected_samples.extend(unit_samples_in_bin)
        
        # order corrected_samples
        corrected_samples = np.sort(corrected_samples)

        # make a corrected dataframe
        corrected_units[u] = pd.DataFrame({'samples': corrected_samples})

    return corrected_units


def main(experiment='robot_single_goal', animal='Rat_HC2', session='15-07-2024'):

    data_dir = get_data_dir(experiment, animal, session)
    
    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    spike_sorting_dir = os.path.join(spike_dir, 'sorter_output')
    units = load_sorted_spikes(spike_sorting_dir)


    # Carla concatenated the files in the wrong order for spike sorting, so load the order she used so times can be 
    # corrected
    with open(os.path.join(spike_sorting_dir, 'spikeinterface_recording.json'), 'r') as f:
        spikeinterface_recording = json.load(f)

    bin_file_list = []
    for r in spikeinterface_recording['kwargs']['recording_list']:
        bin_file = r['kwargs']['recording']['kwargs']['recording']['kwargs']['recording']['kwargs']['folder_path']
        # just keep the bin file name
        bin_file = os.path.basename(bin_file)
        # remove the 'imec0' from the bin file name if it exists
        if '_imec0' in bin_file:
            bin_file = bin_file.replace('_imec0', '')
        bin_file_list.append(bin_file)

    # load the correct order of the bin files
    spikeglx_dir = os.path.join(data_dir, 'spikeglx_data')
    # load spikeglx_n_samples_and_pulses.csv as a dataframe
    n_samples_and_pulses = pd.read_csv(os.path.join(spikeglx_dir, 'spikeglx_n_samples_and_pulses.csv'))
    n_samples_and_pulses['cumulative_samples'] = n_samples_and_pulses['n_samples'].cumsum()

    # make a new data frame using the bin_file_list
    bin_file_df = pd.DataFrame(bin_file_list, columns=['bin_file'])
    # add the n_samples from the n_samples_and_pulses dataframe to the bin_file_df accoring to the bin_file
    bin_file_df['n_samples'] = bin_file_df['bin_file'].map(n_samples_and_pulses.set_index('dir_name')['n_samples'])
    bin_file_df['cumulative_samples'] = bin_file_df['n_samples'].cumsum()


    # correct the spike times for each unit
    corrected_units = correct_spike_times(units, n_samples_and_pulses, bin_file_df)

    save_pickle(corrected_units, 'unit_spike_times', spike_dir)

    pass


if __name__ == "__main__":

    main(experiment='robot_single_goal', animal='Rat_HC2', session='15-07-2024')

    