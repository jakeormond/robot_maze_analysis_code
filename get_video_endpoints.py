import os
import glob
import pandas as pd
import numpy as np
from numpy import matlib as mb
import re

from matplotlib import pyplot as plt
import pickle
import math
from get_directories import get_data_dir 
from load_and_save_data import load_pickle, save_pickle

import cv2

def get_video_endpoints(video_dir, user_input=True):
    # get the list of .avi files
    avi_file_list = glob.glob(os.path.join(video_dir, '*.avi'))

    # loop throught the avi files
    end_frame_dictionary = {}
    for avi_file in avi_file_list:

        # print the name of the file
        print(os.path.basename(avi_file))

        # open the file and calculate its frame rate
        cap = cv2.VideoCapture(avi_file)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # calculate the number of frames
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # calculate the duration of the video
        duration = num_frames / fps
        print('duration: {}'.format(duration))

        if user_input:        
            # ask the user for the end time of the video in
            #  minutes:seconds
            end_time = input('Enter the end time of the video in minutes:seconds: ')
        else:
            end_time = ''
        
        if len(end_time) > 0:
            end_time = '0' + end_time
            # convert the end time to seconds
            end_time_seconds = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
            # calculate the end frame
            end_frame = end_time_seconds * fps
            print('end frame: {}'.format(end_frame))

        else:
            end_time_seconds = duration
            end_time_minutes = int(end_time_seconds//60)
            end_time_seconds = int(end_time_seconds%60)
            end_time = '0' + str(end_time_minutes) + ':' + str(end_time_seconds)
            end_frame = num_frames-1

        # create a dictionary with the video name, the end time, and end frame
        avi_file_name = os.path.basename(avi_file)
        end_frame_dict_temp = {'video_name': avi_file_name,
                          'end_time': end_time,
                          'end_frame': end_frame}
        
        # append the dictionary to the list
        trial_time = avi_file_name[-23:-4]
        end_frame_dictionary[trial_time] = end_frame_dict_temp
    
    return end_frame_dictionary

def get_video_startpoints(dlc_data):
    start_frame_dictionary = {}

    for d in dlc_data.keys():

        # calculate frame intervals
        frame_intervals = np.diff(dlc_data[d]['ts'].values)

        # find the mean of the frame intervals
        frame_interval_mean = np.mean(frame_intervals)

        # the start frame is the first frame with a frame interval less than 1.5 times the mode
        start_frame_dictionary[d] = np.where(frame_intervals < 1.5*frame_interval_mean)[0][0]

    return start_frame_dictionary


if __name__ == "__main__":
    animal = 'Rat47'
    session = '08-02-2024'
    data_dir = get_data_dir(animal, session)
    video_dir = os.path.join(data_dir, 'video_files')

    endpoints = get_video_endpoints(video_dir, user_input=False)
    save_pickle(endpoints, 'video_endpoints', video_dir)

    # start_points IS RUN IN process_dlc_data.py NOT HERE!!!!!
    
    # load dlc_data to get startpoints
    # dlc_dir = os.path.join(data_dir, 'deeplabcut')
    # dlc_data = load_pickle('dlc_processed_data', dlc_dir)
    
    # startpoints = get_video_startpoints(dlc_data)
    # save_pickle(startpoints, 'video_startpoints', video_dir)
    
    pass