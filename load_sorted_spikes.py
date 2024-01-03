import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from get_directories import get_data_dir 

sample_freq = 30000

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

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    
    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_sorted_spikes(spike_dir)

    units_file = os.path.join(spike_dir, 'unit_spike_times.pickle')

    with open(units_file, 'wb') as handle:
        pickle.dump(units, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    del units

    with open(units_file, 'rb') as handle:
        units = pickle.load(handle)

    pass

    
  