# Matches pulses emitted by Bonsai to pulses recorded in SpikeGLX.
# Plots the alignment of the pulses from the beginning and ends of each trial
# so the user can confirm that the pulses are correctly aligned before performing
# subsequent analyses.

import os
import glob
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import h5py
from get_directories import get_home_dir, get_data_dir 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from load_and_save_data import load_pickle, save_pickle



def get_spikeglx_pulses(animal, session):
    # get the data directory
    data_dir = get_data_dir(animal, session)

    # get the spikeglx pulses
    spikeglx_dir = os.path.join(data_dir, 'imec_files')
    spikeglx_files = glob.glob(os.path.join(spikeglx_dir, '*xd*.txt'))
    spikeglx_pulse_file = spikeglx_files[0]

    spikeglx_pulses = pd.read_csv(spikeglx_pulse_file, header=None)
    spikeglx_pulses = spikeglx_pulses[0]

    # split spikeglx pulses into trials. Trials will have interpulse intervals greater than 1 second
    # get the indices of the pulses
    spikeglx_pulses = np.array(spikeglx_pulses)
    
    # get the interpulse intervals
    intervals = np.diff(spikeglx_pulses)
    
    # get number of pulses per trial
    intertrial_indices = np.where(intervals > 20)[0]
    
    # get the trial lengths
    # concantenate intertrial_indices[0] and np.diff(intertrial_indices)
    trial_lengths = np.concatenate(([intertrial_indices[0] + 1], np.diff(intertrial_indices), \
                                   [len(spikeglx_pulses) - (intertrial_indices[-1]+1)]))
    
    return trial_lengths


def get_imec_intertrial_intervals(animal, session):
    # get the data directory
    data_dir = get_data_dir(animal, session)
    imec_dir = os.path.join(data_dir, 'imec_files')
    pulses = load_imec_pulses(imec_dir)

    # convert pulses to seconds
    pulses = [pulse/30000 for pulse in pulses]

    # get the interpulse intervals
    intervals = [np.diff(pulse) for pulse in pulses]

    # find intervals greater than interval length
    interval_length = 2
    intertrial_indices = [np.where(interval > interval_length)[0] for interval in intervals]

    # remove any intervals that are consecutive
    intertrial_intervals = np.where(np.diff(intertrial_indices[0]) == 1)[0]
    
    # remove intertrial_indices that are consecutive
    intertrial_indices = [np.delete(intertrial_indices[0], intertrial_intervals + 1)]

    return intertrial_indices
    
def get_bonsai_n_pulses(data_dir):
    # get the bonsai pulses
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    bonsai_files = glob.glob(os.path.join(bonsai_dir, 'pulseTS*.csv'))
    n_trials = len(bonsai_files)

    n_bonsai_pulses = []

    for i, bonsai_file in enumerate(bonsai_files):
        # read the bonsai pulses
        pulses = pd.read_csv(bonsai_file, header=None)
        n_bonsai_pulses.append(len(pulses))

    return n_bonsai_pulses

def match_bonsai_and_imec_pulses_V1(data_dir):
    
    # get the bonsai pulses
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    bonsai_files = glob.glob(os.path.join(bonsai_dir, 'pulseTS*.csv'))
    # n_trials = len(bonsai_files)

    # get the imec pulses
    imec_dir = os.path.join(data_dir, 'imec_files')
    imec_pulses = load_imec_pulses(directory=imec_dir)
    # n_imec_trials is equal to the number of datasets in imec_pulses
    n_imec_trials = len(imec_pulses)

    # n_pulses is a list of the same length as bonsai_files
    n_pulses = {}  # a dictionary containing the number of pulses in each trial}
    pulses = {}  # also a list containing the pulses; will susbsequently be saved as an h5 file
    pulse_file = "pulseTS"
   
    imec_ind = 0

    for i, bonsai_file in enumerate(bonsai_files):
        trial_time = bonsai_file[-23:-4]
        # read the bonsai pulses
        pulses[trial_time] = pd.read_csv(bonsai_file, header=None)
        # name the column 'bonsai_pulses'
        n_pulses[trial_time] = (len(pulses))

        # convert bonsai_pulses to seconds
        bonsai_seconds = pulses[trial_time][0]/1000
        bonsai_seconds = bonsai_seconds - bonsai_seconds[0]
        print('bonsai seconds: ' + str(bonsai_seconds[0:10]))

        first_bonsai_pulse_ind = np.where(np.diff(bonsai_seconds) > 2)[0][-1] + 1

        for j in range(imec_ind, n_imec_trials):
            imec_seconds = imec_pulses[j]/30000
            imec_seconds = imec_seconds - imec_seconds[0]
            print('imec seconds: ' + str(imec_seconds[0:10]))

            # find first pulse
            diff_imec_seconds = np.diff(imec_seconds)
            # first pulse should be the last value that is greater than 1 before values get smaller than 1
            first_imec_pulse_ind = np.where(diff_imec_seconds > 2)[0][-1] + 1
            
            first_pulse_ind = first_imec_pulse_ind - first_bonsai_pulse_ind

            if first_pulse_ind != 0:
                imec_seconds = imec_seconds[first_pulse_ind:]
                imec_seconds = imec_seconds - imec_seconds[0]
                imec_pulses[j] = imec_pulses[j][first_pulse_ind:]
                print('bonsai seconds: ' + str(bonsai_seconds[0:10]))
                print('imec seconds: ' + str(imec_seconds[0:10]))

            n_imec_pulses = len(imec_pulses[j])
            if n_imec_pulses == len(pulses[trial_time]):
                # add the imec pulses to the bonsai pulses dataframe
                pulses[trial_time] = pulses[trial_time].assign(new_column=imec_pulses[j])
                pulses[trial_time].columns = ['bonsai_pulses_ms', 'imec_pulses_samples']
                imec_ind = j + 1
                break

            if j == n_imec_trials - 1:
                # if we get to the last imec trial and haven't found a match, then we need to throw an error
                raise ValueError('No matching imec pulses found for trial ' + str(i))
                print('No matching imec pulses found for trial ' + str(i))
                print('Number of bonsai pulses: ' + str(len(bonsai_pulses)))

    # now save the dataframes to an h5 file
    pulse_file = pulse_file + '.hdf5'
    pulse_file = os.path.join(bonsai_dir, pulse_file)

    with pd.HDFStore(pulse_file) as hdf:
        for key, df in pulses.items():
            hdf.put(key, df)            
                   
    return pulses


def make_dataframe_pulse_numbers(n_pulses_and_samples, data_dir):
    # get the bonsai pulses
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    bonsai_files = glob.glob(os.path.join(bonsai_dir, 'pulseTS*.csv'))
    n_trials = len(bonsai_files)

    # get the imec pulses
    imec_dir = os.path.join(data_dir, 'imec_files')
    imec_pulses = load_imec_pulses(directory=imec_dir)
    # n_imec_trials is equal to the number of datasets in imec_pulses
    n_imec_trials = len(imec_pulses)

    # n_pulses is a list of the same length as bonsai_files
    n_pulses = {}  # a dictionary containing the number of pulses in each trial}
    pulses = {}  # also a list containing the pulses dataframes; will susbsequently be saved as a pickle file
 
    # assert that the we have the same number of bonsai and imec trials
    assert n_trials == n_imec_trials

    trial_times = []
    n_bonsai_pulses = []
    n_imec_pulses = []

    for i, bonsai_file in enumerate(bonsai_files):
        trial_time = bonsai_file[-23:-4]
        trial_times.append(trial_time)

        # read the bonsai pulses
        bonsai_pulses = pd.read_csv(bonsai_file, header=None)
        # name the column 'bonsai_pulses'
        n_bonsai_pulses.append(len(bonsai_pulses))

        imec_trial_pulses = imec_pulses[i]
        n_imec_pulses.append(len(imec_trial_pulses))

    # make a dataframe with trial_time, n_bonsai_pulses, n_imec_pulses
    pulse_numbers = pd.DataFrame({'trial_time': trial_times, 'n_bonsai_pulses': n_bonsai_pulses, 'n_imec_pulses': n_imec_pulses})

    return pulse_numbers




def match_bonsai_and_imec_pulses(n_pulses_and_samples, data_dir):    
  
    # get the bonsai pulses
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    bonsai_files = glob.glob(os.path.join(bonsai_dir, 'pulseTS*.csv'))
    n_trials = len(bonsai_files)

    # get the imec pulses
    imec_dir = os.path.join(data_dir, 'imec_files')
    imec_pulses = load_imec_pulses(directory=imec_dir)
    # n_imec_trials is equal to the number of datasets in imec_pulses
    n_imec_trials = len(imec_pulses)

    # n_pulses is a list of the same length as bonsai_files
    n_pulses = {}  # a dictionary containing the number of pulses in each trial}
    pulses = {}  # also a list containing the pulses dataframes; will susbsequently be saved as a pickle file
 
    # assert that the we have the same number of bonsai and imec trials
    assert n_trials == n_imec_trials
    
    start_sample = 0
    for i, bonsai_file in enumerate(bonsai_files):
       
        trial_time = bonsai_file[-23:-4]
        # read the bonsai pulses
        pulses[trial_time] = pd.read_csv(bonsai_file, header=None)
        # name the column 'bonsai_pulses'
        n_pulses[trial_time] = (len(pulses))

        # convert bonsai_pulses to seconds
        bonsai_seconds = pulses[trial_time][0]/1000
        bonsai_seconds = bonsai_seconds - bonsai_seconds[0]
        print('bonsai seconds: ' + str(bonsai_seconds[0:10]))

        imec_trial_pulses = imec_pulses[i] + start_sample
        
        if len(imec_trial_pulses) != len(pulses[trial_time]):
            bonsai_seconds = pulses[trial_time][0]/1000
            bonsai_seconds = bonsai_seconds - bonsai_seconds[0]
            print('bonsai seconds: ' + str(bonsai_seconds[0:10]))

            imec_seconds = imec_trial_pulses/30000
            imec_seconds = imec_seconds - imec_seconds[0]
            print('imec seconds: ' + str(imec_seconds[0:10]))
                        
            if len(imec_trial_pulses) > len(pulses[trial_time]): 
                
                while len(imec_trial_pulses) > len(pulses[trial_time]):
                
                    n_extra_pulses = len(imec_trial_pulses) - len(pulses[trial_time])

                    diff_imec = np.diff(imec_seconds)
                    diff_bonsai = np.diff(bonsai_seconds)

                    diff_diff = np.abs(diff_imec[:-n_extra_pulses] - diff_bonsai)

                    # find first diff greater than 0.01
                    first_diff = np.where(diff_diff > 0.01)[0]

                    if first_diff.shape[0] == 0:
                        # remove the last pulse from imec_trial_pulses
                        imec_trial_pulses = imec_trial_pulses[:-1]

                    else:
                        first_diff = first_diff[0]
                        # remove the first_diff pulse from imec_trial_pulses
                        imec_trial_pulses = np.delete(imec_trial_pulses, first_diff+1)
                    
                    imec_seconds = imec_trial_pulses/30000

                    # OLD METHOD
                    # subtract bonsai_seconds from imec_seconds, the values should be positive
                    # get the index of the first negative value and append it to indices_to_remove
                    # imec_bonsai_diff = imec_seconds[:-n_extra_pulses] - bonsai_seconds
                    # negative_diff = np.where(imec_bonsai_diff < 0)[0][0]
                   
                    # imec_trial_pulses = np.delete(imec_trial_pulses, negative_diff)
                    # imec_seconds = imec_trial_pulses/30000
                    # imec_seconds = imec_seconds - imec_seconds[0]
            
            elif len(imec_trial_pulses) < len(pulses[trial_time]):
                # for now, just throw an error
                raise ValueError('Number of imec pulses less than number of bonsai pulses for trial ' + str(i))
            
            assert len(imec_trial_pulses) == len(pulses[trial_time])


           
        # add the imec pulses to the bonsai pulses dataframe
        pulses[trial_time] = pulses[trial_time].assign(new_column=imec_trial_pulses)
        pulses[trial_time].columns = ['bonsai_pulses_ms', 'imec_pulses_samples']
        
        start_sample = start_sample + n_pulses_and_samples[i]['n_samples']
    
    # save pulses as a pickle file
    save_pickle(pulses, 'pulses_dataframes', bonsai_dir)
                   
    return pulses

def load_imec_pulses(directory=None): # these are the pulses generated by extract_pulses_from_raw.py
    
    if directory is None:
    # use tkinter to select the file
        root = tk.Tk()
        root.withdraw()
        print('select IMEC/spikeglx pulse file')
        pulse_file = filedialog.askopenfilename()
    
    else:
        # get pulse file name; it ends with 'pulses.hdf5'        
        # pulse_file = glob.glob(os.path.join(directory, '*pulses.hdf5'))[0]
        pulse_file = os.path.join(directory, 'imec_pulses.hdf5')

    with h5py.File(pulse_file, 'r') as hf:
        keys = list(hf.keys())

        # reorder the keys by the integer value after the final 'g'
        keys = sorted(keys, key=lambda x: int(x.split('g')[-1]))

        # pulses = [hf[f'dataset_{i}'][:] for i in range(len(hf.keys()))]

        pulses = [hf[key][:] for key in keys]

    return pulses 

def load_bonsai_pulses(directory=None): # this the h5 file generated by get_bonsai_pulses
    # use tkinter to select the file
    if directory is None:
        root = tk.Tk()
        root.withdraw()
        print('select Bonsai pulse file')
        pulse_file = filedialog.askopenfilename()
    else:
        bonsai_dir = os.path.join(directory, 'video_csv_files')

        # find file that ends with pulses.hdf5
        pulse_file = glob.glob(os.path.join(bonsai_dir, '*pulses.hdf5'))[0]
        pulse_file = os.path.join(bonsai_dir, pulse_file)
        # pulse_file = os.path.join(bonsai_dir, 'pulseTS.hdf5')

    pulses = {}
    with pd.HDFStore(pulse_file) as hdf:
        keys = list(hdf.keys())
        for key in keys:
            key_temp = key[1:]
            pulses[key_temp] = hdf[key]

    return pulses

def plot_pulse_alignment(pulses, data_dir):

    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    # create a directory to save the figures
    fig_dir = os.path.join(bonsai_dir, 'pulse_alignment')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    trial_times = list(pulses.keys())
    for i, t in enumerate(trial_times):

        trial_name = t
        trial_data = pulses[t]
        data_subset = [None]*2
        data_subset[0] = trial_data.iloc[0:10].values
        data_subset[1] = trial_data.iloc[-10:, :].values

        # create a figure for the data with 2 subplots side by side
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))  #`sharey=True` makes the y-axis the same for both plots)
        fig.suptitle(trial_name)

        for j in range(2):
            data_top = data_subset[j][:,0] # bonsai data
            # covert bonsai data to seconds
            data_top = data_top/1000

            data_bottom = data_subset[j][:,1] # spikeglx data
            data_bottom = data_bottom/30000

            if j == 0:
                data_top_initial = data_top[0]
                data_bottom_initial = data_bottom[0]

            data_top = data_top - data_top_initial
            data_bottom = data_bottom - data_bottom_initial

            # plot data_top as blue vertical tick marks, and data_bottom as red tick marks
            ax[j].vlines(data_top, ymin=i, ymax=i+1, color='blue')
            ax[j].vlines(data_bottom, ymin=i+1, ymax=i+2, color='red')
            ax[j].set_xlabel('Time (s)')

            # create legened - blue is bonsai, red is spikeglx
            blue_patch = mpatches.Patch(color='blue', label='Bonsai')
            red_patch = mpatches.Patch(color='red', label='SpikeGLX')
            ax[j].legend(handles=[blue_patch, red_patch])

            # give subplots titles
            if j == 0:
                ax[j].set_title('First 10 pulses')
            else:
                ax[j].set_title('Last 10 pulses')

            
        plt.show()
        # save the figure
        fig_name = trial_name + '.png'
        fig_name = os.path.join(fig_dir, fig_name)
        fig.savefig(fig_name)
        plt.close()


if __name__ == "__main__":
    animal = 'Rat46'
    session = '20-02-2024'

    data_dir = get_data_dir(animal, session)

    # not currently using get_spikeglx_pulses, use extract_pulses_from_raw.py instead
    # spikeglx_pulses = get_spikeglx_pulses(animal, session)

    imec_pulses = load_imec_pulses(os.path.join(data_dir, 'imec_files'))
    n_pulses_and_samples = load_pickle('n_pulses_and_samples.pkl', os.path.join(data_dir, 'imec_files'))

    pulses_df = make_dataframe_pulse_numbers(n_pulses_and_samples, data_dir)
    # print the entire dataframe for visual inspection
    print(pulses_df)

    pulses = match_bonsai_and_imec_pulses(n_pulses_and_samples, data_dir)

    plot_pulse_alignment(pulses, data_dir)
    pass