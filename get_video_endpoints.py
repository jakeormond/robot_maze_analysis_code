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
from process_dlc_data import load_dlc_processed_pickle

import cv2

def get_video_endpoints(video_dir):
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

        # ask the user for the end time of the video in minutes:seconds
        end_time = input('Enter the end time of the video in minutes:seconds: ')
        
        # convert the end time to seconds
        end_time_seconds = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])
        # calculate the end frame
        end_frame = end_time_seconds * fps
        print('end frame: {}'.format(end_frame))

        # create a dictionary with the video name, the end time, and end frame
        avi_file_name = os.path.basename(avi_file)
        end_frame_dict_temp = {'video_name': avi_file_name,
                          'end_time': end_time,
                          'end_frame': end_frame}
        
        # append the dictionary to the list
        trial_time = avi_file_name[-23:-4]
        end_frame_dictionary[trial_time] = end_frame_dict_temp
    
    return end_frame_dictionary

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    video_dir = os.path.join(data_dir, 'video_files')

    endpoints = get_video_endpoints(video_dir)

    # save as pickle file
    pickle_file = os.path.join(video_dir, 'video_endpoints.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(endpoints, f)

    del endpoints

    # load the pickle file
    with open(pickle_file, 'rb') as f:
        endpoints = pickle.load(f)