import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import pickle
import re 
import openpyxl

import sys
if os.name == 'nt':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
else:
    sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')
    sys.path.append('/home/jake/Documents/robot_maze/workstation')

from utilities.get_directories import get_data_dir, reverse_date
from utilities.load_and_save_data import load_pickle, save_pickle

from honeycomb_task.platform_map import Map


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

def get_goals(data_dir):
    behaviour_dir = get_behaviour_dir(data_dir)
    behav_data_by_goal = load_pickle('behaviour_data_by_goal', behaviour_dir)
    goals = list(behav_data_by_goal.keys())
    return goals


def assess_choices(behaviour_df, platform_map):

    correct_choice = []

    # loop through the rows of the dataframe
    for i, row in behaviour_df.iterrows():
        chosen_pos = row['chosen_pos']
        unchosen_pos = row['unchosen_pos']

        chosen_to_goal_dist = platform_map.cartesian_distance(chosen_pos, platform_map.goal_position)
        unchosen_to_goal_dist = platform_map.cartesian_distance(unchosen_pos, platform_map.goal_position)

        if chosen_to_goal_dist < unchosen_to_goal_dist:
            correct_choice.append(1)
        elif chosen_to_goal_dist > unchosen_to_goal_dist:
            correct_choice.append(0)
        else:
            correct_choice.append(-1)

    behaviour_df['correct_choice'] = correct_choice

    return behaviour_df


def assess_choices_all_trials(csv_files, platform_map):
    # load the csv files
    behaviour_data = {}
    for i, f in enumerate(csv_files):
        behaviour_df = load_behaviour_file(f)
        
        behaviour_df  = assess_choices(behaviour_df, platform_map)
        
        trial_time = behaviour_df.name
        behaviour_data[trial_time] = behaviour_df

    return behaviour_data
        


def main(experiment = 'robot_single_goal', animal = 'Rat_HC2', session = '15-07-2024'):
    
    data_dir = get_data_dir(experiment, animal, session)
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
    # behaviour_data_by_goal = \
    #     split_behaviour_data_by_goal(behaviour_data)
    # save_pickle(behaviour_data_by_goal, 'behaviour_data_by_goal', behaviour_dir)

    pass


def get_goal(csv_files):
    goals = []
    for i, f in enumerate(csv_files):
        behaviour_data_temp = load_behaviour_file(f)
        goals.append(behaviour_data_temp['chosen_pos'].iloc[-1])
    
    goal = np.unique(goals)
    if goal.shape[0] != 1:
        raise ValueError(
            f"More than 2 unique goals found. Unique goals: {goal}"
        )
    
    goal = int(goal[0])

    return goal


def save_dict_of_dfs_to_excel(dict_of_dfs, file_path):
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in dict_of_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name)


def main2(experiment = 'robot_single_goal', animal = 'Rat_HC2', session = '15-07-2024'):

    # load the platform map
    map_dir = '/home/jake/Documents/robot_maze/workstation/map_files'
    # map_file = 'platform_map.csv'
    # map = pd.read_csv(os.path.join(map_dir, map_file))
    # load map as an array
    platform_map = Map(directory=map_dir)

    # load behaviour data
    data_dir = get_data_dir(experiment, animal, session)
    behaviour_dir = get_behaviour_dir(data_dir)

    # find csv files in behaviour directory
    csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))

    # get the goal, it is the last entry in the chosen_pos column
    goal = get_goal(csv_files)
    platform_map.set_goal_position(goal)

    # assess the choices
    behaviour_data = assess_choices_all_trials(csv_files, platform_map)
    # save the behaviour data to a pickle file
    save_pickle(behaviour_data, 'behaviour_data', behaviour_dir)

    file_path = os.path.join(behaviour_dir, 'behaviour_data_all_trials.xlsx')
    save_dict_of_dfs_to_excel(behaviour_data, file_path)


def find_continuous_blocks(lst, value):
    blocks = []
    start = None

    for i, v in enumerate(lst):
        if v == value:
            if start is None:
                start = i
        else:
            if start is not None:
                blocks.append((start, i - 1))
                start = None

    if start is not None:
        blocks.append((start, len(lst) - 1))

    return blocks


def time_to_ms(time_str):
    # Split the time string into hours, minutes, and seconds
    h, m, s = map(float, time_str.split(':'))
    
    # Convert hours, minutes, and seconds to milliseconds
    ms = (h * 3600 + m * 60 + s) * 1000
    
    return int(ms)


def main3(experiment = 'robot_single_goal', animal = 'Rat_HC2', session = '15-07-2024'):

    # load behaviour data
    data_dir = get_data_dir(experiment, animal, session)
    behaviour_dir = get_behaviour_dir(data_dir)

    # create new directory to save the csv files with samples column
    samples_dir = os.path.join(behaviour_dir, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    # find csv files in behaviour directory
    csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))

    # get the goal, it is the last entry in the chosen_pos column
    # load the platform map
    map_dir = '/home/jake/Documents/robot_maze/workstation/map_files'
    platform_map = Map(directory=map_dir)
    goal = get_goal(csv_files)
    platform_map.set_goal_position(goal)

    # load dlc data
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_final_data = load_pickle('dlc_final', dlc_dir)

    # load the cropValues data
    crop_vals_dir = os.path.join(data_dir, 'video_csv_files')

    # loop through the csv files
    crop_val_cols = ['x_crop_vals', 'y_crop_vals']
    for i, f in enumerate(csv_files):
        behaviour_data_temp = load_behaviour_file(f)

        assess_choices(behaviour_data_temp, platform_map)

        trial_time = behaviour_data_temp.name
        # get the dlc data for the trial
        dlc_data = dlc_final_data[trial_time]
        # make a list of tuples from the x_crop_vals and y_crop_vals columns
        crop_vals_dlc = list(zip(dlc_data['x_crop_vals'], dlc_data['y_crop_vals']))

        # get the crop values for the trial
        crop_vals_file = f'cropValues_{trial_time}.csv'
        crop_vals_path = os.path.join(crop_vals_dir, crop_vals_file)
        crop_vals = pd.read_csv(crop_vals_path, usecols=[0,1], names=crop_val_cols)
        # create an empty column called 'samples'
        crop_vals['samples'] = np.nan

        # get the crop times for the trial
        crop_times_file = f'cropTS_{trial_time}.csv'
        crop_times_path = os.path.join(crop_vals_dir, crop_times_file)
        crop_times = pd.read_csv(crop_times_path, header=None)

        crop_vals_used = [(crop_vals.iloc[0]['x_crop_vals'], crop_vals.iloc[0]['y_crop_vals'])]
        
        for i in range(1, len(crop_vals)):
            
            current_crop_vals = (crop_vals.iloc[i]['x_crop_vals'], crop_vals.iloc[i]['y_crop_vals'])
            previous_crop_vals = (crop_vals.iloc[i-1]['x_crop_vals'], crop_vals.iloc[i-1]['y_crop_vals'])

            if current_crop_vals == previous_crop_vals:
                continue

            # check how many times current_crop_vals appears in crop_vals_used
            n = crop_vals_used.count(current_crop_vals)
          
            if n == 0:
                # find the index of the first instance of current_crop_vals in crop_vals_dlc
                index = crop_vals_dlc.index(current_crop_vals)
            
            else: 
                # find the index of the first instance of current_crop_vals in the n+1th continuous
                # block of current_crop_vals in crop_vals_dlc
                blocks = find_continuous_blocks(crop_vals_dlc, current_crop_vals)
                index = blocks[n][0]

            crop_vals.at[i, 'samples'] = dlc_data['video_samples'].iloc[index]
            crop_vals_used.append(current_crop_vals)


        # if there are any nan values in the samples column, interpolate their values using 
        # crop_times values
        if crop_vals['samples'].isnull().values.any():
            # get the indices of the nan values
            nan_indices = crop_vals[crop_vals['samples'].isnull()].index
            for i in nan_indices:
                
                if i == 0:
                    time_diff = crop_times.iloc[0] - crop_times.iloc[1]
                    sample_diff = time_diff * 30 # time in ms, sample rate is 30 kHz
                    crop_vals.at[i, 'samples'] = np.round(crop_vals.at[i+1, 'samples'] + sample_diff)
                    continue

                time_diff = crop_times.iloc[i] - crop_times.iloc[i-1]
                sample_diff = time_diff * 30
                crop_vals.at[i, 'samples'] = np.round(crop_vals.at[i-1, 'samples'] + sample_diff)

        # add samples column from crop_vals to the behaviour data, omitting the first value
        samples_list = crop_vals['samples'].tolist()[2:]  # This will be shorter than the original list
        samples_list.extend([np.NaN] * (len(behaviour_data_temp) - len(samples_list)))  # Extend to match the length

        # Assign the list to the DataFrame column
        behaviour_data_temp['samples'] = samples_list

        # calculate time diff between last 2 choices
        last_time = time_to_ms(behaviour_data_temp['choice_time'].iloc[-1])
        second_last_time = time_to_ms(behaviour_data_temp['choice_time'].iloc[-2])

        time_diff = last_time - second_last_time
        sample_diff = time_diff * 30

        # add the time diff to the last sample
        behaviour_data_temp['samples'].iloc[-1] = behaviour_data_temp['samples'].iloc[-2] + sample_diff
        
        # save the behaviour data to a csv file
        behaviour_data_temp.to_csv(os.path.join(samples_dir, f'{trial_time}_samples.csv'), index=False)
        
      
    pass



if __name__ == "__main__":

    main3(experiment='robot_single_goal', animal='Rat_HC2', session='15-07-2024')

    