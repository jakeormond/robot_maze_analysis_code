import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import re
import shutil
import cv2 
from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import save_pickle, load_pickle

# screen platforms is a dictionary where each key is the screen number and each
# value is the number of the platfom that is directly adjacent to the screen
screen_platforms = {1: 12, 2: 18, 3: 246, 4: 240}

# load the platform map
robot_maze_dir = get_robot_maze_directory()
map_path = os.path.join(robot_maze_dir, 'workstation',
            'map_files', 'platform_map.csv')
platform_map = np.genfromtxt(map_path, delimiter=',')
col_dist = np.round(np.cos(np.radians(30)), 3)  # distances between columns
row_dist = 0.5                                  # and rows in platform map

cm_per_pixel = 370/2048 # 370 cm is the y dimension of the arena, 2048 is the y dimension of the video


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


def get_screen_coordinates(data_dir): # these are the 4 tv screens in the corners of the arena, just want the rough centre coordinate

    # get the coordinates of the points clicked by the user
    screen_coordinates_temp = []
    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            screen_coordinates_temp.append([x, y])
            # Draw a small circle at the clicked position
            cv2.circle(fs_frame_og, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('frame', fs_frame_og)
            print(screen_coordinates_temp)

    # get path to full_video.avi, which is two directories above data_dir
    full_video_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), "full_video_w_screen_platforms.avi")
    cap = cv2.VideoCapture(full_video_path)

    while True:
        # Read the frame       
        _, fs_frame_og = cap.read() 
        # if frame is not completely black, break the loop
        if np.sum(fs_frame_og) > 0:
            break

    # resize the frame to half its size
    fs_frame_og = cv2.resize(fs_frame_og, (0, 0), fx=0.5, fy=0.5)

    # Create a named window
    cv2.namedWindow('frame')

    # Set the mouse callback for the window to your get_coordinates function
    cv2.setMouseCallback('frame', get_coordinates)

    # Display the frame in the window
    cv2.imshow('frame', fs_frame_og)

    # Wait for a key press and then close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap.release()

    # adjust the coordinates to account for the fact that the frame was resized
    screen_coordinates_temp = np.array(screen_coordinates_temp)
    screen_coordinates_temp = screen_coordinates_temp * 2

    screen_coordinates = {}
    # convert the coordinates to a dictionary
    for i in range(4):
        screen_coordinates[i+1] = screen_coordinates_temp[i,:]

    return screen_coordinates


def get_goal_coordinates(platform_coordinates=None, goals=None, data_dir=None):

    if data_dir is None and platform_coordinates is None:
        raise ValueError('Must provide either data_dir or platform_coordinates')

    if platform_coordinates is None:
        robot_maze_dir = get_robot_maze_directory()
        platform_path = os.path.join(robot_maze_dir, 'workstation',
                'map_files', 'platform_coordinates.pickle')
        with open(platform_path, 'rb') as f:
            platform_coordinates = pickle.load(f)

    if goals is None:

        behaviour_dir = os.path.join(data_dir, 'behaviour')
        behaviour_pickle_path = os.path.join(behaviour_dir, 
                'behaviour_data_by_goal.pkl')
        with open(behaviour_pickle_path, 'rb') as f:
            behaviour_data = pickle.load(f)
        goals = []
        for k in behaviour_data.keys():
            goals.append(k)


    goal_coordinates = {}
    for g in goals:
        goal_coordinates[g] = platform_coordinates[g][0:2]
    return goal_coordinates

def get_relative_head_direction(dlc_data, platform_coordinates, goals, screen_coordinates):
    
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

        # calculate the hd relative to each goal
        for g in goals:
            relative_direction_temp = dlc_data[d]['hd'] - dlc_data[d][f'goal_direction_{g}']
            # any relative direction greater than pi is actually less than pi
            relative_direction_temp[relative_direction_temp > np.pi] -= 2*np.pi
            # any relative direction less than -pi is actually greater than -pi
            relative_direction_temp[relative_direction_temp < -np.pi] += 2*np.pi
            dlc_data[d][f'relative_direction_{g}'] = relative_direction_temp

        # calculate the direction to each screen
        for s in screen_coordinates.keys():
            x_diff = screen_coordinates[s][0] - x
            y_diff = y - screen_coordinates[s][1]
            dlc_data[d][f'screen_direction_{s}'] = np.arctan2(y_diff, x_diff)

        # calculate the hd relative to each screen
        for s in screen_coordinates.keys():
            relative_direction_temp = dlc_data[d]['hd'] - dlc_data[d][f'screen_direction_{s}']
            # any relative direction greater than pi is actually less than pi
            relative_direction_temp[relative_direction_temp > np.pi] -= 2*np.pi
            # any relative direction less than -pi is actually greater than -pi
            relative_direction_temp[relative_direction_temp < -np.pi] += 2*np.pi
            dlc_data[d][f'relative_direction_screen{s}'] = relative_direction_temp
       
    return dlc_data, goal_coordinates


def get_distances(dlc_data, platform_coordinates, goal_coordinates, screen_coordinates): # get distance to each goal and screen. Distances will be calculated
    
    # get the goal indices
    goal_indices = {}
    for g in goal_coordinates.keys():
        # find the row and col coordinates of the goal in the platform map array
        row, col = np.where(platform_map == g)
        goal_indices[g] = [row[0], col[0]]    

    # get the screen indices
    screen_indices = {}
    for s in screen_coordinates.keys():
        # find the row and col coordinates of the screen in the platform map array
        row, col = np.where(platform_map == screen_platforms[s])
        screen_indices[s] = [row[0], col[0]]
    
    # in both pixels and in platform distances (to account for the effect of the camera lens)
    for d in dlc_data.keys():
        # get the x and y coordinates of the animal's head
        x = dlc_data[d]['x'].values
        y = dlc_data[d]['y'].values

        # get the platform positions for each frame
        occupied_platforms = dlc_data[d]['occupied_platforms'].values
        # platform_indices is a 2d array where each row is the row and col. 
        # Its shape is (len(occupied_platforms), 2)
        platform_indices = np.zeros((len(occupied_platforms), 2))

        # find the row and col coordinates of the occupied platforms in the platform map array
        for i, p in enumerate(occupied_platforms):
            # find the row and col coordinates of the occupied platform in the platform map array
            row, col = np.where(platform_map == p)
            platform_indices[i, 0] = row[0]
            platform_indices[i, 1] = col[0]

        # goal distances
        for g in goal_coordinates.keys():            
            # calculate the distance to each goal in pixels
            x_diff = goal_coordinates[g][0] - x
            y_diff = goal_coordinates[g][1] - y
            dlc_data[d][f'distance_to_goal_{g}'] = np.sqrt(x_diff**2 + y_diff**2)

            # calculate the distance to each goal in platform distances
            row_diff = goal_indices[g][0] - platform_indices[:,0]
            col_diff = goal_indices[g][1] - platform_indices[:,1]

            row_diff = row_diff * row_dist
            col_diff = col_diff * col_dist

            dlc_data[d][f'distance_to_goal_{g}_platform'] = np.sqrt(row_diff**2 + col_diff**2)

        # calculate the distance to each screen
        for s in screen_coordinates.keys():
            # calculate the distance to each screen in pixels
            x_diff = screen_coordinates[s][0] - x
            y_diff = screen_coordinates[s][1] - y
            dlc_data[d][f'distance_to_screen_{s}'] = np.sqrt(x_diff**2 + y_diff**2)

            # calculate the distance to each screen in platform distances
            row_diff = screen_indices[s][0] - platform_indices[:,0]
            col_diff = screen_indices[s][1] - platform_indices[:,1]

            row_diff = row_diff * row_dist
            col_diff = col_diff * col_dist

            dlc_data[d][f'distance_to_screen_{s}_platform'] = np.sqrt(row_diff**2 + col_diff**2)

    return dlc_data


def get_x_and_y_limits(dlc_data):

    x_and_y_limits = {}

    min_x = 100000
    max_x = -100000

    min_y = 100000
    max_y = -100000

    for t in dlc_data.keys():
        x = dlc_data[t]['x'].values
        y = dlc_data[t]['y'].values

        min_x = np.min([min_x, np.min(x)])
        max_x = np.max([max_x, np.max(x)])

        min_y = np.min([min_y, np.min(y)])
        max_y = np.max([max_y, np.max(y)])
    
    x_and_y_limits['x'] = [min_x, max_x]
    x_and_y_limits['y'] = [min_y, max_y]

    return x_and_y_limits

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    dlc_dir = os.path.join(data_dir, 'deeplabcut')

    # get the goal coordinates
    screen_coordinates = get_screen_coordinates(data_dir)
    # save the screen coordinates
    save_pickle(screen_coordinates, 'screen_coordinates', dlc_dir)
  
    # load dlc_data which has the trial times    
    dlc_data = load_pickle('dlc_final', dlc_dir)
    
    # load the platform coordinates, from which we can get the goal coordinates
    robot_maze_dir = get_robot_maze_directory()
    platform_path = os.path.join(robot_maze_dir, 'workstation', 'map_files')
    platform_coordinates = load_pickle('platform_coordinates', platform_path)
    crop_coordinates = load_pickle('crop_coordinates', platform_path)

    platform_coordinates, save_flag = get_uncropped_platform_coordinates(platform_coordinates, crop_coordinates)
    if save_flag:
        # first, copy the original platform_coordinates file
        src_file = os.path.join(robot_maze_dir, 'workstation',
                'map_files', 'platform_coordinates.pickle')
        dst_file = os.path.join(robot_maze_dir, 'workstation',
                'map_files', 'platform_coordinates_cropped.pickle')
        shutil.copyfile(src_file, dst_file)     

        # save the new platform_coordinates file
        save_pickle(platform_coordinates, 'platform_coordinates', platform_path)

    # load the behaviour data, from which we can get the goal ids
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    behaviour_data = load_pickle('behaviour_data_by_goal', behaviour_dir)

    goals = []
    for k in behaviour_data.keys():
        goals.append(k)

    # calculate the animal's current platform for each frame
    dlc_data = get_current_platform(dlc_data, platform_coordinates)  

    # calculate head direction relative to the goals
    dlc_data, goal_coordinates = get_relative_head_direction(dlc_data, platform_coordinates, goals, screen_coordinates)

    # calculate the distance to each goal and screen
    dlc_data = get_distances(dlc_data, platform_coordinates, goal_coordinates, screen_coordinates)

    # save the dlc_data
    save_pickle(dlc_data, 'dlc_final', dlc_dir)
    