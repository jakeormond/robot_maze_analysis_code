import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from utilities.get_directories import get_data_dir
from utilities.load_and_save_data import load_pickle, save_pickle
from spikes.restrict_spikes_to_trials import concatenate_unit_across_trials
from position.calculate_occupancy import get_direction_bins
from spikes.calculate_consinks import find_consink
from spikes.calculate_spike_pos_hd import circularly_translate_units_by_goal

cm_per_pixel = 0.2




















def calculate_translated_mrl(unit, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks):
    translated_unit = circularly_translate_units_by_goal(unit, dlc_data)
    mrl, _, _ = find_consink(translated_unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks)
    return mrl


def recalculate_consink_to_all_candidates_from_translation(unit, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks):

    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)

    mrl = Parallel(n_jobs=-1, verbose=50)(delayed(calculate_translated_mrl)(unit, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles))

    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    ci = (mrl_95, mrl_999)
    
    return ci


def calculate_shift_mrl(hd, min_shift, max_shift, unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks):
    shift = np.random.randint(min_shift, max_shift)
    hd_shift = np.roll(hd, shift)
    shifted_unit = unit.copy()
    shifted_unit['hd'] = hd_shift

    mrl, _, _ = find_consink(shifted_unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks)
    return mrl


def recalculate_consink_to_all_candidates_from_shuffle(unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks):

    hd = unit['hd'].to_numpy()

    # calculate min and max numbers of shifts
    min_shift = len(hd)//15
    max_shift = len(hd) - min_shift

    n_shuffles = 1000
    mrl = np.zeros(n_shuffles)

    mrl = Parallel(n_jobs=-1, verbose=50)(delayed(calculate_shift_mrl)(hd, min_shift, max_shift, unit, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks) for s in range(n_shuffles))

    mrl = np.round(mrl, 3)
    mrl_95 = np.percentile(mrl, 95)
    mrl_999 = np.percentile(mrl, 99.9)

    ci = (mrl_95, mrl_999)
    
    return ci


def main():
    animal = 'Rat_HC1'
    session = '31-07-2024'

    data_dir = os.path.join('/ceph/scratch/jakeo/robot_maze/single_goal_expt', animal, session)
    spike_dir = os.path.join(data_dir, 'spike_sorting')

    neuron_types = load_pickle('neuron_types', spike_dir)

    # get direction bins
    direction_bins = get_direction_bins(n_bins=12)

    # load positional data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data_concat_by_goal = load_pickle('dlc_data_concat_by_goal', dlc_dir)

    units = load_pickle('units_by_goal', spike_dir)

    goals = units.keys()

    reldir_occ_by_pos = np.load(os.path.join(dlc_dir, 'reldir_occ_by_pos.npy'))
    sink_bins = load_pickle('sink_bins', dlc_dir)
    candidate_sinks = load_pickle('candidate_sinks', dlc_dir)

    consinks_df = load_pickle('consinks_df', spike_dir)

    for goal in goals:
        goal_units = units[goal]

        dlc_data = dlc_data_concat_by_goal[goal][['video_samples', 'x', 'y', 'hd']]

        # make columns for the confidence intervals; place them directly beside the mrl column
        # if the columns don't exist, insert them            
        if 'ci_95' not in consinks_df[goal].columns:
            idx = consinks_df[goal].columns.get_loc('mrl')
            consinks_df[goal].insert(idx + 1, 'ci_95', np.nan)
            consinks_df[goal].insert(idx + 2, 'ci_999', np.nan)

        for cluster in goal_units.keys():

            if cluster not in neuron_types.keys() or neuron_types[cluster] != 'pyramidal':
                continue
           
            unit = concatenate_unit_across_trials(goal_units[cluster])
            unit = unit[['samples', 'x', 'y', 'hd']]

            ######### PERFORM CICULAR TRANSLATION CONTROL

            print(f'calcualting confidence intervals for goal {goal} {cluster}')

            ci = recalculate_consink_to_all_candidates_from_translation(unit, dlc_data, reldir_occ_by_pos, sink_bins, direction_bins, candidate_sinks)
            # ci = recalculate_consink_to_all_candidates_from_shuffle(unit, reldir_occ_by_pos, sink_bins,  direction_bins, candidate_sinks)
            
            consinks_df[goal].loc[cluster, 'ci_95'] = ci[0]
            consinks_df[goal].loc[cluster, 'ci_999'] = ci[1]

    save_pickle(consinks_df, 'consinks_df_translated_ctrl', spike_dir)
    # save as csv
    # consinks_df.to_csv(os.path.join(spike_dir, 'consinks_df_translated_ctrl.csv'))
    print('saved consinks_df_translated_ctrl to {spike_dir}')    

if __name__ == "__main__":

    main()

   
   
