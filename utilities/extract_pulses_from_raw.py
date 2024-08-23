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

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_home_dir, get_data_dir 
from utilities.load_and_save_data import load_pickle, save_pickle

def get_imec_dirs(dir = None):
    if dir is None:
        # use tkinter to select the directory   
        print('select spikeglx bin directory')
        root = tk.Tk()
        root.withdraw()
        imec_path = filedialog.askdirectory()
    else:
        imec_path = dir


    # list the folders in this directory
    imec_dirs = [f for f in os.listdir(imec_path) if os.path.isdir(os.path.join(imec_path, f))]

    # for each folder, get the final part of the path, which is a number after the final "g"
    imec_nums = [int(re.findall(r'\d+$', f)[0]) for f in imec_dirs]

    # sort the folders by the imec number
    imec_dirs = [x for _, x in sorted(zip(imec_nums, imec_dirs))]

    # get the full path of each folder
    imec_dirs = [os.path.join(imec_path, f) for f in imec_dirs]

    return imec_dirs


def get_bin_meta_paths(dir = None):

    if dir is None:
        # use tkinter to select the directory   
        print('select spikeglx bin directory')
        root = tk.Tk()
        root.withdraw()
        bin_path = filedialog.askopenfilename()
    else:
        bin_path = dir

    # get the directory name
    bin_dir = os.path.basename(bin_path)
    # within the directory, there is another directory that begins with the same name as bin_dir
    # this is the directory that contains the meta file
    imec_dir = glob.glob(os.path.join(bin_path, os.path.basename(bin_dir) + '*'))[0]
   
    # get the meta file
    meta_file = glob.glob(os.path.join(imec_dir, '*.meta'))[0]

    # get the bin file
    bin_file = glob.glob(os.path.join(imec_dir, '*.bin'))[0]

    return bin_file, meta_file


def get_n_spikeglx_samples(meta_path):
    meta_data = open(meta_path, 'r')
    sample_rate = 30000

    while True:
        text_line = meta_data.readline()

        if text_line.find('fileTimeSecs') != -1: # found the line
            eq_ind = text_line.find('=')
            slash_ind = text_line.find('\\')
            seconds = float(text_line[eq_ind+1:slash_ind])
            n_samples = int(np.round(seconds * sample_rate))
            meta_data.close()
            break

    meta_data = open(meta_path, 'r')
    while True:
        text_line = meta_data.readline()

        if text_line.find('fileSizeBytes') != -1: # found the line
            eq_ind = text_line.find('=')
            slash_ind = text_line.find('\\')
            bytes = float(text_line[eq_ind+1:slash_ind])
            n_channels = 385
            n_samples2 = int(bytes/(n_channels*2)) # divided by 2 because each value is 2 bytes (i.e. 16 bit)
            break

    assert n_samples == n_samples2, 'n_samples and n_samples2 are not equal'  

    return n_samples


def extract_spikeglx_pulses(bin_path, meta_path):
    
    meta_data = open(meta_path, 'r')
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

    # get rid of weird values in digital_data
    low_ind = np.where(digital_data < 64)[0]
    digital_data[low_ind] = 0

    high_ind = np.where(digital_data > 64)[0]
    digital_data[high_ind] = 64

    # get pulse onsets 
    pulse_onsets = np.where(np.diff(digital_data) > 50)[0] + 1
    n_onsets = np.shape(pulse_onsets)[0]

    # get pulse offsets
    pulse_offsets = np.where(np.diff(digital_data) < 0)[0] 
    n_offsets = np.shape(pulse_offsets)[0]

    # assert that the number of onsets and offsets are equal
    assert n_onsets == n_offsets, 'number of onsets and offsets are not equal'
    
    return pulse_onsets, n_samples


def extract_spikeglx_cat_pulses(bin_path, meta_path):
    
    meta_data = open(meta_path, 'r')
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
    pulse_onsets = np.where(np.diff(digital_data) > 50)[0] + 1
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

    # file needs to be saved one directory up
    save_path = os.path.join(os.path.dirname(bin_path), pulse_file)

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


def sort_key(x):
    # Extract the time part (e.g., '14h24')
    time_part = re.search(r'(\d{1,2}h\d{2})', x).group(1)
    # Convert time part to a sortable format (e.g., '14h24' -> '1424')
    time_sortable = int(time_part.replace('h', ''))
    
    # Extract the digit after the final 'g' (e.g., '12')
    digit_part = int(re.search(r'g(\d+)$', x).group(1))
    
    return (time_sortable, digit_part)


def main(experiment = 'robot_single_goal', animal = 'Rat_HC2', session = '15-07-2024'):

    data_dir = get_data_dir(experiment, animal, session)

    spikeglx_dir = os.path.join(data_dir, 'spikeglx_data')    

    # list the folders within spikeglx_dir
    spikeglx_data_dirs = [f for f in os.listdir(spikeglx_dir) if os.path.isdir(os.path.join(spikeglx_dir, f))]

    # order the folders by the integer after the final g
    spikeglx_data_dirs = sorted(spikeglx_data_dirs, key=sort_key)

    # save the order of the folders as a csv file
    spikeglx_data_dirs_df = pd.DataFrame(spikeglx_data_dirs)
    spikeglx_data_dirs_df.to_csv(os.path.join(spikeglx_dir, 'spikeglx_data_dirs.csv'), header=False, index=False)
    
    # create lists to store the names of the pulses, and the number of samples
    pulses = []
    n_pulses = []
    n_samples = []

    # loop through each folder and extract the pulses
    for i, d in enumerate(spikeglx_data_dirs):

        print(f'index is {i}, extracting pulses from ' + d)

        # there is another folder within this folder that contains the bin and meta files
        spikeglx_data_dir = os.path.join(spikeglx_dir, d)
        bin_file, meta_file = get_bin_meta_paths(dir = spikeglx_data_dir)

        n_samples.append(get_n_spikeglx_samples(meta_file))

        pulses_temp, _ = extract_spikeglx_pulses(bin_file, meta_file)
        n_pulses.append(pulses_temp.shape[0])
        pulses.append(pulses_temp)

        print(f'number of pulses: {n_pulses[i]}')


    # create h5 file to store the pulses in spikeglx_dir
    pulse_file = os.path.join(spikeglx_dir, 'spikeglx_pulses.hdf5')
    with h5py.File(pulse_file, 'w') as hf:
        print('saving pulses to ' + pulse_file)
        for i, arr in enumerate(pulses):
            hf.create_dataset(spikeglx_data_dirs[i], data=arr)
            print(f'saving {arr.shape[0]} pulses to key {spikeglx_data_dirs[i]}')

    # create a dataframe to store the number of samples and pulses
    data = {
        'dir_name': spikeglx_data_dirs,
        'n_samples': n_samples,
        'n_pulses': n_pulses
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(spikeglx_dir, 'spikeglx_n_samples_and_pulses.csv'), index=False)

        # load the pulses from the h5 file
    print('loading pulses from h5 file')
    with h5py.File(pulse_file, 'r') as hf:
        # get keys
        keys = list(hf.keys())
        print(f'keys in h5 file: {keys}')
        
        # loop through the keys and print the number of pulses
        for key in keys:
            print(f'{key}: {hf[key].shape[0]} pulses from h5 file')

            # get the n_samples and n_pulses from df and print them
            data_dict = df.set_index('dir_name').T.to_dict()
            print(f'{key}: {data_dict[key]["n_pulses"]} pulses from dataframe')
            print(f'{key}: {data_dict[key]["n_samples"]} samples from dataframe')

    pass






if __name__ == "__main__":

    main(experiment = 'robot_single_goal', animal = 'Rat_HC2', session = '15-07-2024')
    pass

    
    