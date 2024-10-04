import os
import glob
import pandas as pd
import numpy as np
from numpy import matlib as mb
import re
from matplotlib import pyplot as plt
import pickle
import math
import imageio
import cv2

import sys

if os.name == 'nt':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
else:
    sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')

from utilities.get_directories import get_data_dir 
from utilities.load_and_save_data import load_pickle, save_pickle
from position.calculate_pos_and_dir import get_goal_coordinates


def create_cropped_video_with_dlc_data(dlc_data, 
                        video_path, start_and_end):
    arrow_len = 10
    cap = cv2.VideoCapture(video_path)

    # properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # create video_with_tracking folder
    video_name = os.path.basename(video_path)
    video_with_tracking_dir = os.path.join(os.path.dirname(video_path), \
                                           'cropped_videos_with_tracking')
    if not os.path.exists(video_with_tracking_dir):
        os.mkdir(video_with_tracking_dir)

    # create video writer
    output_video_path = os.path.join(video_with_tracking_dir, \
                                     video_name)
    # video_writer = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), 
                               fps, (height, width))
    
    num_frames = dlc_data.shape[0]

    for frame_idx in range(num_frames):
        ret, frame = cap.read()

        # get the dlc data for this frame
        dlc_for_frame = dlc_data.iloc[frame_idx, :]
       
        hd_for_frame = dlc_for_frame['hd']

        if math.isnan(hd_for_frame):
            continue

        x1 = np.cos(hd_for_frame) * arrow_len
        y1 = -np.sin(hd_for_frame) * arrow_len

        x2 = np.cos(hd_for_frame + np.pi) * arrow_len
        y2 = -np.sin(hd_for_frame + np.pi) * arrow_len

        x_cropped = dlc_for_frame['x_cropped']
        y_cropped = dlc_for_frame['y_cropped']        
        
        arrow_start = (int(x_cropped + x1), 
                       int(y_cropped + y1))
        arrow_end = (int(x_cropped + x2), 
                     int(y_cropped + y2))
        
        cv2.arrowedLine(frame, arrow_end, arrow_start, 
                        (0, 0, 255), 4, tipLength = 0.5)
        
        if frame_idx < start_and_end[0]:
            # write BEFORE START on the frame along the top
            cv2.putText(frame, 'BEFORE START', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        if frame_idx > start_and_end[1]:
            # write END on the frame along the top
            cv2.putText(frame, 'END', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


        cv2.startWindowThread()

        cv2.imshow('Video Player', frame)

        video_writer.write(frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()  

def create_full_video_with_dlc_data(video_time, dlc_data, data_dir, start_and_end):

    # get path to full_video.avi, which is two directories above data_dir
    full_video_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), "full_video.avi")
    # cap = cv2.VideoCapture(full_video_path)
    # _, fs_frame_og = cap.read()
    # display the frame to the user
    # cv2.imshow('Video Player', frame)

    # fs_frame_og is a 2048 x 2448 x 3 array of uint8 and is black
    fs_frame_og = np.zeros((2048, 2448, 3))
        
    # get video path
    video_dir = os.path.join(data_dir, 'video_files')

    # get goal coordinates
    goal_coordinates = get_goal_coordinates(data_dir=data_dir)
    
    # blank full size frame 
    # blank_fs_frame = np.zeros((2048, 2448, 3))

    arrow_len = 20
    video_path = os.path.join(video_dir, 'video_' + video_time + '.avi')
    cap = cv2.VideoCapture(video_path)

    # get number of frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # properties
    height = 1024
    width = 1224
    fps = 10

    # create video_with_tracking folder
    video_name = os.path.basename(video_path)
    video_with_tracking_dir = os.path.join(os.path.dirname(video_path), \
                                           'full_videos_with_tracking')
    if not os.path.exists(video_with_tracking_dir):
        os.mkdir(video_with_tracking_dir)

    # create video writer
    output_video_path = os.path.join(video_with_tracking_dir, \
                                     video_name)
    # video_writer = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc(*'XVID'), 
                               fps, (width, height), isColor=True)
    
    # num_frames = dlc_data.shape[0]

    for frame_idx in range(num_frames):
        _, frame = cap.read()

        if frame_idx < start_and_end[0]:
            continue

        if frame_idx > start_and_end[1]:
            break

        if frame_idx % 10 != 0:
            continue

        # get the row from dlc_data with index frame_idx
        dlc_for_frame = dlc_data.loc[frame_idx, :]       
        # dlc_for_frame = dlc_data.iloc[frame_idx, :]
        
        hd_for_frame = dlc_for_frame[ 'hd']

        if math.isnan(hd_for_frame):
            continue

        # x_crop_pos = int(dlc_for_frame['x'] 
        #                  - dlc_for_frame['x_cropped'])
        # y_crop_pos = int(dlc_for_frame['y'] 
        #                  - dlc_for_frame['y_cropped'])
        
        x_crop_pos = int(dlc_for_frame['x_crop_vals'])
        y_crop_pos = int(dlc_for_frame['y_crop_vals'])
       
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # fs_frame = blank_fs_frame
        fs_frame = fs_frame_og.copy()
        # fs_frame[y_crop_pos:y_crop_pos + 600, x_crop_pos:x_crop_pos + 600] = gray_frame
        fs_frame[y_crop_pos:y_crop_pos + frame.shape[0], x_crop_pos:x_crop_pos + frame.shape[1]] = frame
        fs_frame = fs_frame.astype('uint8') 

        x1 = np.cos(hd_for_frame) * 1.5 * arrow_len
        y1 = -np.sin(hd_for_frame) * 1.5 * arrow_len

        x2 = np.cos(hd_for_frame + np.pi) * 1.5 * arrow_len
        y2 = -np.sin(hd_for_frame + np.pi) * 1.5 * arrow_len

        x = dlc_for_frame['x']
        y = dlc_for_frame['y']        
        
        arrow_start = (int(x + x1), 
                       int(y + y1))
        arrow_end = (int(x + x2), 
                     int(y + y2))
        
        cv2.arrowedLine(fs_frame, arrow_end, arrow_start, 
                        (0, 0, 255), 8, tipLength = 0.5)
                
        # draw goals
        radius = 80
        colours = [(255, 0, 0), (0, 255, 0)]


        # if goal_coordinates is a dictionary
        if type(goal_coordinates) == dict:

            for i, g in enumerate(goal_coordinates.keys()):
                x_goal = goal_coordinates[g][0]
                y_goal = goal_coordinates[g][1]
                cv2.circle(fs_frame, (int(x_goal), int(y_goal)), 
                        int(radius), colours[i], 8)
            
                # draw goal direction arrow
                goal_dir = dlc_for_frame[f'goal_direction_{g}']
                x1 = np.cos(goal_dir) * 4*arrow_len
                y1 = -np.sin(goal_dir) * 4*arrow_len

                # x2 = np.cos(goal_dir + np.pi) * arrow_len
                # y2 = -np.sin(goal_dir + np.pi) * arrow_len

                x2 = np.cos(goal_dir) * arrow_len
                y2 = -np.sin(goal_dir) * arrow_len

                arrow_start = (int(x + x1), 
                        int(y + y1))
                arrow_end = (int(x + x2), 
                            int(y + y2))
                
                cv2.arrowedLine(fs_frame, arrow_end, arrow_start, 
                                colours[i], 8, tipLength = 0.5)
        
                # get the relative direction to the goal
                relative_dir = dlc_for_frame[f'relative_direction_{g}']
                
                # plot the relative direction as the animal's head direction
                # relative to the goal, which is plotted above the arrow
                circle_radius = 80

                # if i is 0, plot in the upper left corner, otherwise plot in the
                # upper right corner
                if i == 0:
                    x_offset = 4*circle_radius
                else:
                    x_offset = 2448 - 4*circle_radius
                
                y_offset = 2*circle_radius

                # draw the circle
                circle_centre = (int(circle_radius/2) + x_offset, 
                                int(circle_radius/2) + y_offset)
                cv2.circle(fs_frame, circle_centre, circle_radius, colours[i], 8)

                # overlay the text "goal" along the top of the circle
                cv2.putText(fs_frame, 'GOAL', (int(x_offset - 0.5*circle_radius), 
                                            int(y_offset - 0.75*circle_radius)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, colours[i], 2, cv2.LINE_AA)

                # draw the arrow
                arrow_len2 = 0.5*circle_radius
                x1 = np.cos(relative_dir + np.pi/2) * arrow_len2
                y1 = -np.sin(relative_dir + np.pi/2) * arrow_len2

                x2 = np.cos(relative_dir + (3/2)*np.pi) * arrow_len2
                y2 = -np.sin(relative_dir + (3/2)*np.pi) * arrow_len2

                arrow_start = (int((circle_radius/2) + x_offset + x1), 
                            int((circle_radius/2) + 2*y_offset + y1))
                arrow_end = (int((circle_radius/2) + x_offset + x2),  
                            int((circle_radius/2) + 2*y_offset + y2))

                cv2.arrowedLine(fs_frame, arrow_end, arrow_start, 
                                [0, 0, 255], 8, tipLength = 0.5)

            # add x and y coordinates and hd_for_frame as text along the bottom of the frame
            cv2.putText(fs_frame, f'x: {x:.2f}, y: {y:.2f}, hd: {hd_for_frame:.2f}, rd2g: {relative_dir:.2f}',
                        (100, fs_frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)

            # shrink the frame to keep file size under control
            fs_frame_re = cv2.resize(fs_frame, (1224, 1024))

            cv2.startWindowThread()

            cv2.imshow('Video Player', fs_frame_re)

            video_writer.write(fs_frame_re)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break 

        else:
            i = 0 # only one goal, so select the first colour for the arrow
            x_goal = goal_coordinates[0]
            y_goal = goal_coordinates[1]
            cv2.circle(fs_frame, (int(x_goal), int(y_goal)), 
                    int(radius), colours[i], 8)
        
            # draw goal direction arrow
            goal_dir = dlc_for_frame[f'goal_direction']
            x1 = np.cos(goal_dir) * 4*arrow_len
            y1 = -np.sin(goal_dir) * 4*arrow_len

            # x2 = np.cos(goal_dir + np.pi) * arrow_len
            # y2 = -np.sin(goal_dir + np.pi) * arrow_len

            x2 = np.cos(goal_dir) * arrow_len
            y2 = -np.sin(goal_dir) * arrow_len

            arrow_start = (int(x + x1), 
                    int(y + y1))
            arrow_end = (int(x + x2), 
                        int(y + y2))
            
            cv2.arrowedLine(fs_frame, arrow_end, arrow_start, 
                            colours[i], 8, tipLength = 0.5)
    
            # get the relative direction to the goal
            relative_dir = dlc_for_frame[f'relative_direction_to_goal']
            
            # plot the relative direction as the animal's head direction
            # relative to the goal, which is plotted above the arrow
            circle_radius = 80

            # if i is 0, plot in the upper left corner, otherwise plot in the
            # upper right corner
            if i == 0:
                x_offset = 4*circle_radius
            else:
                x_offset = 2448 - 4*circle_radius
            
            y_offset = 2*circle_radius

            # draw the circle
            circle_centre = (int(circle_radius/2) + x_offset, 
                            int(circle_radius/2) + y_offset)
            cv2.circle(fs_frame, circle_centre, circle_radius, colours[i], 8)

            # overlay the text "goal" along the top of the circle
            cv2.putText(fs_frame, 'GOAL', (int(x_offset - 0.5*circle_radius), 
                                        int(y_offset - 0.75*circle_radius)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, colours[i], 2, cv2.LINE_AA)

            # draw the arrow
            arrow_len2 = 0.5*circle_radius
            x1 = np.cos(relative_dir + np.pi/2) * arrow_len2
            y1 = -np.sin(relative_dir + np.pi/2) * arrow_len2

            x2 = np.cos(relative_dir + (3/2)*np.pi) * arrow_len2
            y2 = -np.sin(relative_dir + (3/2)*np.pi) * arrow_len2

            arrow_start = (int((circle_radius/2) + x_offset + x1), 
                        int((circle_radius/2) + 2*y_offset + y1))
            arrow_end = (int((circle_radius/2) + x_offset + x2),  
                        int((circle_radius/2) + 2*y_offset + y2))

            cv2.arrowedLine(fs_frame, arrow_end, arrow_start, 
                            [0, 0, 255], 8, tipLength = 0.5)

            # add x and y coordinates and hd_for_frame as text along the bottom of the frame
            cv2.putText(fs_frame, f'x: {x:.2f}, y: {y:.2f}, hd: {hd_for_frame:.2f}, rd2g: {relative_dir:.2f}', 
                        (100, fs_frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8, cv2.LINE_AA)

            # shrink the frame to keep file size under control
            fs_frame_re = cv2.resize(fs_frame, (1224, 1024))

            cv2.startWindowThread()

            cv2.imshow('Video Player', fs_frame_re)

            video_writer.write(fs_frame_re)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break 
       
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()


def create_gif_from_video(video_path, gif_path, start_and_end_time):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    
    frame_number = 0
    while True:
        _, frame = cap.read()
        
        # calculate the time of this frame
        frame_time = frame_number/fps

        # if the frame time is before the start time, skip this frame
        if frame_time < start_and_end_time[0] or frame_number % 20 != 0:
            frame_number += 1
            continue
        elif frame_time > start_and_end_time[1]:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_number += 1
    
    cap.release()
    imageio.mimsave(gif_path, frames, 'GIF', fps=fps/10, loop=0)

def get_video_paths_from_dlc(dlc_processed_data, data_dir):
    video_dir = os.path.join(data_dir, 'video_files')
    video_paths = {}
    for d in dlc_processed_data.keys():
        video_time = d
        video_path = os.path.join(video_dir, "video_" + video_time + ".avi")
        video_paths[video_time] = video_path        
        
    return video_paths 


def main(experiment='robot_single_goal', animal='Rat_HC2', session='15-07-2024', n_trials = 2):

    data_dir = get_data_dir(experiment, animal, session)

    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    
    dlc_processed_data = load_pickle('dlc_processed_data', dlc_dir)

    dlc_final_data = load_pickle('dlc_final', dlc_dir)
    
    video_paths = get_video_paths_from_dlc(dlc_processed_data, data_dir)

    video_dir = os.path.join(data_dir, 'video_files')
   
    # load video_startpoints.pkl
    video_startpoints = load_pickle('video_startpoints', video_dir)
 
    # load video_endpoints.pkl
    video_endpoints = load_pickle('video_endpoints', video_dir)

    for i, d in enumerate(dlc_processed_data.keys()):

        if i == n_trials:
            break

        video_time = d
        print(video_time)

        # find the correct video path for this video time
        video_path = video_paths[video_time]
        
        # get video startpoint and endpoint
        video_startpoint = video_startpoints[video_time]
        video_endpoint = video_endpoints[video_time]['end_frame']
        start_and_end = (video_startpoint, video_endpoint)

        # create the video with the dlc data
        # create_cropped_video_with_dlc_data(dlc_processed_data[d], 
        #                                video_path, start_and_end)
            
        create_full_video_with_dlc_data(video_time, dlc_final_data[d], 
                                               data_dir, start_and_end)

    
    # create a gif from video "video_2023-11-08_16.52.26.avi" using frames
    # from 70 seconds to 97 seconds
    # video_path = os.path.join(video_dir, "cropped_videos_with_tracking", "video_2023-11-08_16.52.26.avi")
    # gif_path = os.path.join(video_dir, "cropped_videos_with_tracking", "video_2023-11-08_16.52.26.gif")
    # start_time = 70
    # end_time = 90
    # create_gif_from_video(video_path, gif_path, (start_time, end_time))


    pass


if __name__ == "__main__":
    
    main(experiment='robot_single_goal', animal='Rat_HC2', session='15-07-2024')

    