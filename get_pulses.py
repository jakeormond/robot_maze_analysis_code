# matches pulses emitted by Bonsai to pulses recorded in SpikeGLX

import os
import glob
import numpy as np
import pandas as pd
import pickle
import re

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
    trial_lengths = np.concatenate(([intertrial_indices[0] + 1], np.diff(intertrial_indices)))
    
    return trial_lengths
    
def get_bonsai_pulses(animal, session):
    # get the data directory
    data_dir = get_data_dir(animal, session)

    # get the bonsai pulses
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    bonsai_files = glob.glob(os.path.join(bonsai_dir, 'pulseTS*.csv'))

    # n_pulses is a list of the same length as bonsai_files
    n_pulses = []
    
    for bonsai_file in bonsai_files:
        pulses = pd.read_csv(bonsai_file, header=None)
        n_pulses.append(len(pulses))
    
    aborted_dir = os.path.join(bonsai_dir, 'aborted_trials')
    aborted_files = glob.glob(os.path.join(aborted_dir, 'pulseTS*.csv'))
    aborted_pulses = []

    for aborted_file in aborted_files:
        pulses = pd.read_csv(aborted_file, header=None)
        aborted_pulses.append(len(pulses))

    
    return n_pulses, aborted_pulses

def get_home_dir():
    # determine operating system
    if os.name == 'nt':
        home_dir = 'D:/analysis' # WINDOWS
    elif os.name == 'posix': # Linux or Mac OS
        home_dir = "/media/jake/LaCie" # Linux/Ubuntu
    return home_dir

def get_data_dir(animal, session):
    home_dir = get_home_dir()
    data_dir = os.path.join(home_dir, animal, session)
    return data_dir


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    spikeglx_pulses = get_spikeglx_pulses(animal, session)
    bonsai_pulses, aborted_pulses = get_bonsai_pulses(animal, session)
    pass