import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
import shutil
from get_directories import get_data_dir, get_robot_maze_directory


def get_uncropped_platform_coordinates(platform_coordinates, crop_coordinates):

    # check if you need to run this function
    x = []
    y = []
    for p in platform_coordinates.keys():
        x.append(platform_coordinates[p][0])
        y.append(platform_coordinates[p][1])
    
    x = np.max(x)
    y = np.max(y)

    if x > 1000 and y > 1000: # if the coordinates are already uncropped
        save_flag = False
        return platform_coordinates, save_flag

    # the coordinates have not yet been uncropped
    platforms = list(platform_coordinates.keys())
    crop_keys = list(crop_coordinates.keys())
    # plat_crop_keys is a dictionary, where each key is a platform number and
    # each value is the crop key for its entry in crop_coordinates
    uncropped_platform_coordinates = {}
    plat_crop_keys = {}
    for p in platforms:

        plat_coor_cropped = platform_coordinates[p]

        for k in crop_keys:            
            # extract the integers from k. There may be as many as 3 integers
            k_ints = [int(s) for s in re.findall(r'\d+', k)]

            # determine if integer p is one of the integers in k_ints
            if p in k_ints:
                # if it is, then k is the crop key for platform p
                plat_crop_keys[p] = k
                crop_coor = crop_coordinates[k]

                center = (plat_coor_cropped[0] + crop_coor[0], 
                      plat_coor_cropped[1] + crop_coor[1])
                radius = plat_coor_cropped[2]

                uncropped_platform_coordinates[p] = [center[0], center[1], radius]
                break    

    save_flag = True
    return uncropped_platform_coordinates, save_flag

def get_current_platform(dlc_data, platform_coordinates):

    # create list of platforms and array of coordinates
    platforms = list(platform_coordinates.keys())
    coordinate_array = np.array(list(platform_coordinates.values()))
    coordinate_array = coordinate_array[:,0:2]

    for d in dlc_data.keys():
        
        # get all the x and y positions
        x = dlc_data[d]['x_body'].values
        y = dlc_data[d]['y_body'].values

        # calculate the distance between the animal and each platform for each frame
        # first, copy x and y into 2d arrays with shape len(x) x len(platforms)
        x = np.tile(x, (len(platforms), 1))
        y = np.tile(y, (len(platforms), 1))
        
        # subtract the x and y coordinates of each platform from the x and y coordinates
        # of the animal
        x_diff = x - coordinate_array[:,0].reshape(len(platforms), 1)
        y_diff = y - coordinate_array[:,1].reshape(len(platforms), 1)
        distance = np.sqrt(x_diff**2 + y_diff**2)

        # find the row index of the minimum distance for each frame (i.e. column)
        min_distance_index = np.argmin(distance, axis=0)
        occupied_platforms = np.array(platforms)[min_distance_index]
        
        # add the occupied platforms to the dlc_temp dataframe
        dlc_data[d]['occupied_platforms'] = occupied_platforms

    return dlc_data


def get_relative_head_direction(dlc_data, platform_coordinates, goals):
    
    # get the goal coordinates
    goal_coordinates = {}
    for g in goals:
        goal_coordinates[g] = platform_coordinates[g][0:2]
    
    for d in dlc_data.keys():
        # get the x and y coordinates of the animal's head
        x = dlc_data[d]['x'].values
        y = dlc_data[d]['y'].values
              
        # calculate the direction to each goal
        for g in goals:
            x_diff = goal_coordinates[g][0] - x
            y_diff = y - goal_coordinates[g][1]
            dlc_data[d][f'goal_direction_{g}'] = np.arctan2(y_diff, x_diff)

        # add the goal directions to dlc_data[d]
        for g in goals:
            relative_direction_temp = dlc_data[d]['hd'] - dlc_data[d][f'goal_direction_{g}']
            # any relative direction greater than pi is actually less than pi
            relative_direction_temp[relative_direction_temp > np.pi] -= 2*np.pi
            # any relative direction less than -pi is actually greater than -pi
            relative_direction_temp[relative_direction_temp < -np.pi] += 2*np.pi
            dlc_data[d][f'relative_direction_{g}'] = relative_direction_temp
       
    return dlc_data


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)

    # load dlc_data which has the trial times
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_final.pkl')
    with open(dlc_pickle_path, 'rb') as f:
        dlc_data = pickle.load(f)

    # load the platform coordinates, from which we can get the goal coordinates
    robot_maze_dir = get_robot_maze_directory()
    platform_path = os.path.join(robot_maze_dir, 'workstation',
            'map_files', 'platform_coordinates.pickle')
    with open(platform_path, 'rb') as f:
        platform_coordinates = pickle.load(f)
     
    crop_path = os.path.join(robot_maze_dir, 'workstation', 
            'map_files', 'crop_coordinates.pickle')
    with open(crop_path, 'rb') as f:
        crop_coordinates = pickle.load(f)

    platform_coordinates, save_flag = get_uncropped_platform_coordinates(platform_coordinates, crop_coordinates)
    if save_flag:
        # first, copy the original platform_coordinates file
        src_file = os.path.join(robot_maze_dir, 'workstation',
                'map_files', 'platform_coordinates.pickle')
        dst_file = os.path.join(robot_maze_dir, 'workstation',
                'map_files', 'platform_coordinates_cropped.pickle')
        shutil.copyfile(src_file, dst_file)     

        # save the new platform_coordinates file
        with open(platform_path, 'wb') as f:
            pickle.dump(platform_coordinates, f, protocol=pickle.HIGHEST_PROTOCOL)  


    # load the behaviour data, from which we can get the goal ids
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    behaviour_pickle_path = os.path.join(behaviour_dir, 
            'behaviour_data_by_goal.pkl')
    with open(behaviour_pickle_path, 'rb') as f:
        behaviour_data = pickle.load(f)

    goals = []
    for k in behaviour_data.keys():
        goals.append(k)

    # calculate the animal's current platform for each frame
    dlc_data = get_current_platform(dlc_data, platform_coordinates)  

    # calculate head direction relative to the goals
    dlc_data = get_relative_head_direction(dlc_data, platform_coordinates, goals)

    pass