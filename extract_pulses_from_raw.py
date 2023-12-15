# extract pulses from spikeglx bin file
#
import os
import glob
import numpy as np
import pandas as pd
import pickle
import re
import tkinter as tk
from tkinter import filedialog
import h5py


def extract_spikeglx_pulses():
    # use tkinter to select the file
    root = tk.Tk()
    root.withdraw()
    bin_path = filedialog.askopenfilename()

    # also get the meta file
    meta_file = filedialog.askopenfilename()
    meta_data = open(meta_file, 'r')
    sample_rate = 30000

    while True:
        text_line = meta_data.readline()

        if text_line.find('fileTimeSecs') != -1: # found the line
            eq_ind = text_line.find('=')
            slash_ind = text_line.find('\\')
            seconds = float(text_line[eq_ind+1:slash_ind])
            n_samples = int(seconds * sample_rate)
            break
    
    meta_data.close()
    
    n_channels = 385
    window_size_last = 300000 # 10 seconds   
    window_size_next = window_size_last
    offset_mult = 0
    n_samples_used = 0

    # digital_data is a 1d array of shape (n_samples,)
    digital_data = np.zeros(n_samples, dtype=np.int16)

    while True:
        n_samples_left = n_samples - n_samples_used
        if n_samples_left == 0:
            break
        elif n_samples_left < window_size_last:
            window_size_next = n_samples_left

        # read in the data
        raw_data = np.memmap(bin_path, np.int16, mode='r', shape=(window_size_next, n_channels),
            offset = int(window_size_last * offset_mult) * n_channels * 2) # multiplied by 2 because each value is 2 bytes (i.e. 16 bit)
        raw_data = raw_data.T # shape becomes n_channels X samples

        digital_channel = raw_data[384, -window_size_next:]

        digital_data[int(window_size_last * offset_mult) : int(window_size_last * offset_mult) \
                     + window_size_next] = digital_channel
        
        n_samples_used = int(window_size_last * offset_mult) + window_size_next
        offset_mult += 1


    # get pulse onsets 
    pulse_onsets = np.where(np.diff(digital_data) > 0)[0] + 1
    n_onsets = np.shape(pulse_onsets)[0]

    # get pulse offsets
    pulse_offsets = np.where(np.diff(digital_data) < 0)[0] 
    n_offsets = np.shape(pulse_offsets)[0]

    # assert that the number of onsets and offsets are equal
    assert n_onsets == n_offsets, 'number of onsets and offsets are not equal'
    n_pulses_total = n_onsets

    # get the pulse widths
    pulse_widths = pulse_offsets - pulse_onsets
    long_pulse_ind = np.where(pulse_widths > 100)[0]
    print('number of long pulses: ' + str(np.shape(long_pulse_ind)[0]))

    # get pulse intervals
    pulse_intervals = np.diff(pulse_onsets)
    pulse_intervals = pulse_intervals / sample_rate # convert to seconds
    long_interval_ind = np.where(pulse_intervals > 20)[0]

    n_trials = np.shape(long_interval_ind)[0] + 1
    # make an empty list of length n_trials    
    pulses = [None]*n_trials
    for i in range(n_trials):
        if i == 0 and n_trials == 1:
            pulses[0] = pulse_onsets
        elif i == 0:
            pulses[0] = pulse_onsets[:long_interval_ind[i]+1]
        elif i == n_trials - 1:
            pulses[i] = pulse_onsets[long_interval_ind[i-1]+1:]
        else:
            pulses[i] = pulse_onsets[long_interval_ind[i-1]+1:long_interval_ind[i]+1]

    # get the name of the bin file from bin_path
    bin_file = os.path.basename(bin_path)
    # remove everything from the first period
    bin_file = bin_file.split('.')[0]
    # add on "pulses" to the end
    pulse_file = bin_file + '_pulses'

    # save the pulses to an hdf5 file
    pulse_file = pulse_file + '.hdf5'
    pulse_file = os.path.join(os.path.dirname(bin_path), pulse_file)
    
    with h5py.File(pulse_file, 'w') as hf:
        for i, arr in enumerate(pulses):
            hf.create_dataset(f'dataset_{i}', data=arr)
    
    pulse_file_lengths = [len(pulse) for pulse in pulses]
    print('number of pulses in each trial: ' + str(pulse_file_lengths))
    # save pulse_file_lengths to a csv file
    pulse_file_lengths = np.array(pulse_file_lengths)
    pulse_file_lengths = pd.DataFrame(pulse_file_lengths)
    pulse_file_lengths.to_csv(os.path.join(os.path.dirname(bin_path), 'pulse_file_lengths.csv'), header=False, index=False)

    return pulses

def load_pulses():
    # use tkinter to select the file
    root = tk.Tk()
    root.withdraw()
    pulse_file = filedialog.askopenfilename()

    with h5py.File(pulse_file, 'r') as hf:
        # get keys
        keys = list(hf.keys())
        pulses = [hf[key][:] for key in keys]

    return pulses   


if __name__ == "__main__":
    # pulses = extract_spikeglx_pulses()
    # del pulses
    pulses = load_pulses()
    pass