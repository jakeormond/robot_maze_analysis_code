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
from load_and_save_data import save_pickle, load_pickle

sample_rate = 30000
upsample_factor = 10 # for upsampling the waveform to get the halfwidth

def calculate_mean_rates(units, dlc_data):
    
    session_duration = 0
    for t in dlc_data.keys():
        trial_duration = (dlc_data[t].video_samples.iloc[-1] - 
                dlc_data[t].video_samples.iloc[0])/sample_rate
        session_duration += trial_duration

    mean_rates = {}
    for u in units.keys():
        n_spikes = 0
        for t in units[u].keys():
            # add the number of spikes in units[u][t] to n_spikes
            n_spikes += len(units[u][t])
        mean_rates[u] = np.round(n_spikes/session_duration, 3)
    
    return mean_rates


def get_average_waveforms(units, spike_dir, bin_path):
    n_spikes = 100
    n_samples = 60
    n_channels = 385 # 384 channels + 1 digital channel

    avg_waveforms = {}
    peak_channel = {} 
    channels = {} # all channels on which the cluster features

    # load the template ids of each spike
    spike_templates = np.load(os.path.join(spike_dir, 
                               'spike_templates.npy'))

    spike_clusters = np.load(os.path.join(spike_dir, 
                                'spike_clusters.npy'))
    
    # load the templates. Each template consists of the 
    # spike waveform across all channels. 
    # The shape is (n_templates, n_samples, n_channels). 
    # For some reason n_channels is 383 - need to investigate. 
    templates = np.load(os.path.join(spike_dir, 
                                'templates.npy'))

    for u in units.keys():
       
        unit = units[u]
        # cluster number is integer after 'cluster_' in u
        cluster = int(u.split('_')[1])
        
        # find spikes that belong to this cluster
        spike_ind = np.nonzero(spike_clusters == cluster)[0]

        # get template most commonly associated with this particular unit
        # templates_for_cluster, _ = stats.mode(spike_templates[spike_ind], 
        #                                      keepdims=False)[0]
        templates_for_cluster = np.unique(spike_templates[spike_ind])       
        if templates_for_cluster.shape[0] != 1:
            # use the most common template, i.e. the mode
            templates_for_cluster = stats.mode(spike_templates[spike_ind], 
                                                 keepdims=False)[0]
            
        
        # determine on which channel this template has its largest amplitude. 
        template_across_channels = np.squeeze(templates[templates_for_cluster, :, :]).T
        max_val_all_ch = np.max(np.abs(template_across_channels), axis=1)
        cluster_channel = np.argmax(max_val_all_ch)
        max_ind = np.where(np.abs(template_across_channels[cluster_channel,:]) 
                        == np.max(max_val_all_ch))
        peak_channel[u] = cluster_channel 

        # also, determine all the channels where template features significantly, as we
        # can use this to focus our wavelet analysis on only those channels with spikes
        max_across_channels = np.squeeze(np.abs(template_across_channels[:, max_ind]))    
        mean_val = np.mean(max_across_channels)
        std_val = np.std(max_across_channels)
        channels[u] = np.where(max_across_channels > mean_val + 2*std_val)  

        # select 100 spikes at random
        spikes = []
        for t in unit.keys():
            spikes.append(unit[t].values)
        
        spikes = np.concatenate(spikes)
        if spikes.shape[0] > n_spikes:
            spikes = sample(list(spikes), n_spikes)

        # get the waveform for each spike
        waveform_data = np.zeros((n_spikes, n_samples))

        for i,s in enumerate(spikes):
            # data_temp = np.memmap(full_path, np.int16, mode='r', shape=(60, n_channels),
            # offset=10 * n_channels * 2)

            data_temp = np.memmap(bin_path, np.int16, mode='r', shape=(60, n_channels),
                offset = int(s-30) * n_channels * 2) # multiplied by 2 because each value is 2 bytes (i.e. 16 bit)

            waveform_data[i,:] = data_temp[:, cluster_channel].T

        avg_waveforms[u] = np.mean(waveform_data, axis=0)

    return avg_waveforms

def plot_average_waveforms(average_waveforms, mean_rates, spike_dir):

    # create folder if it doesn't exist
    if not os.path.exists(os.path.join(spike_dir, 'average_waveforms')):
        os.makedirs(os.path.join(spike_dir, 'average_waveforms'))

    for u in average_waveforms.keys():

        # calculate spike width
        halfwidth, halfwidth_ind, waveform_upsampled = \
            calculate_spike_halfwidth(average_waveforms[u])

        # create plot
        # create a new figure
        # plt.figure()
        fig, ax = plt.subplots()
        plt.plot(waveform_upsampled)

        # draw a line from the rising edge to the falling edge
        plt.plot(halfwidth_ind, [waveform_upsampled[halfwidth_ind[0]], 
                                 waveform_upsampled[halfwidth_ind[1]]], 
                                 color='red')

        # write text with halfwidth and mean firing rate
        ax.text(0.05, 0.5, f'Halfwidth: {halfwidth} ms\nMean firing rate: {mean_rates[u]} Hz', 
                transform=ax.transAxes)        

        plt.title(u)
        plt.xlabel('Samples * upsample_factor')
        plt.ylabel('Amplitude')
        plt.savefig(os.path.join(spike_dir, 'average_waveforms', u + '.png'))
        plt.close()


def calculate_spike_halfwidths(average_waveforms):
    
    halfwidths = {}
    for u in average_waveforms.keys():

        # calculate spike width
        halfwidth, _, _ = \
            calculate_spike_halfwidth(average_waveforms[u])
        halfwidths[u] = halfwidth

    return halfwidths

def calculate_spike_halfwidth(average_waveform):
    
    positive_peak = np.max(average_waveform)
    negative_peak = np.min(average_waveform)

    if np.abs(negative_peak) > positive_peak:
        peak = negative_peak
    else:
        peak = -positive_peak
        # flip the waveform so code still works
        average_waveform = -average_waveform
    
    baseline = np.mean(average_waveform[0:20])
    
    waveform_len = average_waveform.shape[0]
    waveform_upsampled = np.interp(np.linspace(0, 
        waveform_len, waveform_len*upsample_factor), np.arange(waveform_len), 
        average_waveform)
    upsampled_len = waveform_upsampled.shape[0]

    # rising edge is value closest to half-peak in first half of waveform
    denominator = 4 # set this to 2 for halfwidth, 4 for quarterwidth
    half_peak = baseline + (peak-baseline)/denominator
   
    if half_peak < 0:
        rising_edge = np.where(waveform_upsampled < half_peak)[0][0]
    else:
        rising_edge = np.where(waveform_upsampled > half_peak)[0][0]
        
    # falling edge is value closest to half-peak in second half of waveform
    if rising_edge < upsampled_len/2:
        middle_ind = int(upsampled_len/2)
    else:
        middle_ind = rising_edge + 3*upsample_factor

    if half_peak < 0:
        falling_edge = np.where(waveform_upsampled[middle_ind:] > half_peak)[0][0] + middle_ind

    else:
        falling_edge = np.where(waveform_upsampled[middle_ind:] > half_peak)[0][0] + middle_ind

    halfwidth = falling_edge - rising_edge
    halfwidth = ((halfwidth/upsample_factor)/sample_rate) * 1000 # in ms
    halfwidth = np.round(halfwidth, 3)

    halfwidth_ind = [rising_edge, falling_edge]

    if np.abs(negative_peak) < positive_peak:
        # flip the waveform back
        waveform_upsampled = -waveform_upsampled

    return halfwidth, halfwidth_ind, waveform_upsampled


def remove_bad_clusters(units, average_waveforms, mean_rates):
    """
    Manually remove any clusters that don't look like neurons.
    """
    # ask the user if there are any bad clusters; the user should enter
    # a 'y' or 'n'
    bad_clusters = input('Are there any bad clusters? (y/n): ')
    if bad_clusters == 'n':
        save_flag = False
        return units, average_waveforms, mean_rates, save_flag

    save_flag = True
    # get user input in a loop, so they can enter the cluster number 
    # one at a time. When they're done, they hit enter with no input
    bad_clusters = []
    while True:
        bad_cluster = input('Enter bad cluster: ')
        if bad_cluster == '':
            break
        else:
            bad_clusters.append(bad_cluster)
    
    # remove bad clusters
    for b in bad_clusters:
        del units[f'cluster_{b}']
        del average_waveforms[f'cluster_{b}']
        del mean_rates[f'cluster_{b}']

    return units, average_waveforms, mean_rates, save_flag

          
def classify_neurons(halfwidths, mean_rates):
    # not currently ussing mean_rates
    neuron_types = {}

    # get user input for halfwidth cutoff and mean rate cutoff
    halfwidth_cutoff = float(input('Enter halfwidth cutoff (ms): '))
    mean_rate_cutoff = float(input('Enter mean firing rate cutoff (Hz): '))

    for u in halfwidths.keys():
        if halfwidths[u] <= halfwidth_cutoff or mean_rates[u] >= mean_rate_cutoff:
            neuron_types[u] = 'interneuron'
        else:
            neuron_types[u] = 'pyramidal'
    
    return neuron_types


if __name__ == "__main__":
    animal = 'Rat65'
    session = '10-11-2023'
    data_dir = get_data_dir(animal, session)

    # load dlc_data which has the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_data = load_pickle('dlc_final', dlc_dir)

    # load the spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    units = load_pickle('restricted_units', spike_dir)
    
    # hardcoding the directory of the neuropixel bin file, as
    # it's not in the data directory
    bin_dir = '/media/jake/LaCie/' + animal + '/' + session
    bin_path = glob.glob(bin_dir + '/*.ap.bin')[0]

    # calculate mean firing rates
    mean_rates = calculate_mean_rates(units, dlc_data)

    # get average waveforms
    # average_waveforms = get_average_waveforms(units, spike_dir, bin_path)
    # save_pickle(average_waveforms, 'average_waveforms', spike_dir)

    # plot average waveforms
    average_waveforms = load_pickle('average_waveforms', spike_dir)
    # plot_average_waveforms(average_waveforms, mean_rates, spike_dir)

    # manually remove any clusters that don't look like neurons
    # units, average_waveforms, mean_rates, save_flag = \
    #     remove_bad_clusters(units, average_waveforms, mean_rates)

    # if save_flag is True:
    #     # resave units and average_waveforms
    #     save_pickle(units, 'restricted_units', spike_dir)
    #     save_pickle(average_waveforms, 'average_waveforms', spike_dir)

    # get halfwidths    
    halfwidths = calculate_spike_halfwidths(average_waveforms)

    # plot halfwidths vs mean firing rates
    plt.figure()
    for u in mean_rates.keys():
        plt.scatter(mean_rates[u], halfwidths[u])
    plt.xlabel('Mean firing rate (Hz)')
    plt.ylabel('Halfwidth (ms)')
    plt.savefig(os.path.join(spike_dir, 'average_waveforms', 
                             'halfwidth_vs_mean_firing_rate.png'))
    # classify neurons
    neuron_types = classify_neurons(halfwidths, mean_rates)
    save_pickle(neuron_types, 'neuron_types', spike_dir)
    

    pass