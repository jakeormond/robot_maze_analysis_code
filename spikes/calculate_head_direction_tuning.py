import os
import numpy as np
import pandas as pd
from utilities.get_directories import get_data_dir
from utilities.load_and_save_data import load_pickle, save_pickle
from position.calculate_occupancy import get_direction_bins, get_directional_occupancy_from_dlc, concatenate_dlc_data, bin_directions
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials


import pycircstat as pycs

if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    # get direction bins
    direction_bins = get_direction_bins(n_bins=12)
    dir_bin_centres = (direction_bins[1:] + direction_bins[:-1])/2

    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)
    dlc_data_concat = concatenate_dlc_data(dlc_data)
    directional_occupancy = get_directional_occupancy_from_dlc(dlc_data_concat, n_bins=12)  

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('units_by_goal', spike_dir)

    goals = units.keys()
    
    hd_tuning = {}
    hd_tuning_df = {}
    for goal in goals:
        goal_units = units[goal]
        hd_tuning[goal] = {}

        for cluster in goal_units.keys():
            unit = concatenate_unit_across_trials(goal_units[cluster])
            head_directions = unit['spike_directions']

            counts, bin_indices = bin_directions(head_directions, direction_bins)

            binned_rates = counts/directional_occupancy

            mrl = pycs.resultant_vector_length(dir_bin_centres, w=binned_rates)
            mean_angle = pycs.mean(dir_bin_centres, w=binned_rates)
            pval, z = pycs.rayleigh(dir_bin_centres, w=None)

            hd_tuning[goal][cluster] = {'mrl': mrl, 'mean_angle': mean_angle, 'pval': pval, 'z': z}
        
        hd_tuning_df[goal] = pd.DataFrame(hd_tuning[goal]).T

    save_pickle(hd_tuning_df, 'hd_tuning', spike_dir)





