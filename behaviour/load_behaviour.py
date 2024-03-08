import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import pickle
import re 

import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir, reverse_date
from utilities.load_and_save_data import load_pickle, save_pickle


def get_behaviour_dir(data_dir):
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    return behaviour_dir

def get_behaviour_file_path(animal, session, time):
    """
    Get the behaviour file name. 
    
    Parameters
    ----------
    animal : str
        The animal name.
    session : str
        The session name.
    time : str
        The time of the behaviour file.
        
    Returns
    -------
    behaviour_file : str
        The behaviour file name.
    """
    # animal = 'Rat64'
     #session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    behaviour_file = f'{reverse_date(session)}_{time}.csv'
    behaviour_file_path = os.path.join(behaviour_dir, behaviour_file)
    return behaviour_file_path

def load_behaviour_file(behaviour_file_path, time=None):
                        
    if time is None:
        filename = os.path.basename(behaviour_file_path)
        # daytime is the last 8 characters of the filename
        # before the .csv
        time = filename[0:-4]

    behaviour_data = pd.read_csv(behaviour_file_path)
    behaviour_data.name = time
    
    return behaviour_data

def split_behaviour_data_by_goal(behaviour_data):
    """
    Split the behaviour data by goal.
    
    Parameters
    ----------
    behaviour_data : pandas dataframe
        The behaviour data.
        
    Returns
    -------
    behaviour_data_by_goal : dict
        The behaviour data split by goal. The keys are the goal names and the 
        values are the behaviour data for each goal.
    """

    goals = []
    behaviour_data_by_goal = {}

    for b in behaviour_data.keys():
        # the goal is the final entry in the chosen_pos column
        data_temp = behaviour_data[b]
        goal = int(data_temp['chosen_pos'].iloc[-1])
        goals.append(goal)

        # check if goal is a key in behaviour_data_by_goal
        if goal not in behaviour_data_by_goal.keys():
            behaviour_data_by_goal[goal] = {}
        
        # add the behaviour data to the goal
        behaviour_data_by_goal[goal][b] = data_temp

    # get the unique goals
    unique_goals = np.unique(goals)
    
    if len(unique_goals) != 2:
        raise ValueError(
            f"More than 2 unique goals found. Unique goals: {unique_goals}"
        )
    
    return behaviour_data_by_goal


def save_behav_data(behav_data, behav_dir):
    # if behav_data is a list
    if type(behav_data) == list:
        trial_times = []
        goals = []
    
        for b in behav_data:
            trial_times.append(b.name)
            goals.append(b.goal)

        data_to_save = {'data': behav_data, 
                        'trial_times': trial_times, 'goals': goals}
        
        filename = 'behaviour_data.pkl'
    
    # if behav_data is a dict
    elif type(behav_data) == dict:
        # get the keys 
        keys = list(behav_data.keys())
        data_to_save = {}
        
        for k in keys:
            trial_times = []

            for b in behav_data[k]:
                trial_times.append(b.name)

            data_to_save[k] = {'data': behav_data[k], 
                        'trial_times': trial_times}
        
        filename = 'behaviour_data_by_goal.pkl'

    # save the data
    pickle_path = os.path.join(behav_dir, filename)
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_to_save, f)   

    return pickle_path 

def load_behav_data(pickle_path):
    with open(pickle_path, 'rb') as f:
        behaviour_temp = pickle.load(f)
        keys = list(behaviour_temp.keys())
        # if the first key begins with 'goal'
        if re.match('goal', keys[0]): # loop through the 2 goals
            behaviour_data = {}
            for k in keys:
                behaviour_data[k] = behaviour_temp[k]['data']
                
                for i, b in enumerate(behaviour_data[k]):
                    b.name = behaviour_temp[k]['trial_times'][i]
        
        else:
            behaviour_data = behaviour_temp['data']
            for i, b in enumerate(behaviour_data):
                b.name = behaviour_temp['trial_times'][i]
                b.goal = behaviour_temp['goals'][i]
    
    return behaviour_data

def split_dictionary_by_goal(dictionary, data_dir):
    behaviour_dir = get_behaviour_dir(data_dir)
    behav_data_by_goal = load_pickle('behaviour_data_by_goal', behaviour_dir)

    split_dictionary = {}
    for g in behav_data_by_goal.keys():
        split_dictionary[g] = {}
        for t in behav_data_by_goal[g].keys():
            split_dictionary[g][t] = dictionary[t]
        
    return split_dictionary 


if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = get_behaviour_dir(data_dir)

    # find csv files in behaviour directory
    csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))

    # load the csv files
    behaviour_data = {}
    for i, f in enumerate(csv_files):
        behaviour_data_temp = load_behaviour_file(f)
        trial_time = behaviour_data_temp.name
        behaviour_data[trial_time] = behaviour_data_temp

    # save the behaviour data to a pickle file
    save_pickle(behaviour_data, 'behaviour_data', behaviour_dir)
       
    # split the behaviour data by goal and save to a pickle file
    behaviour_data_by_goal = \
        split_behaviour_data_by_goal(behaviour_data)
    save_pickle(behaviour_data_by_goal, 'behaviour_data_by_goal', behaviour_dir)


    pass