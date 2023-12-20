import cv2
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


def create_video_with_dlc_data(dlc_data, video_path):
    arrow_len = 10
    cap = cv2.VideoCapture(video_path)

    # properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # create video_with_tracking folder
    base_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_with_tracking_dir = os.path.join(os.path.dirname(video_path), \
                                           'videos_with_tracking')
    if not os.path.exists(video_with_tracking_dir):
        os.mkdir(video_with_tracking_dir)

    # create video writer
    output_video_path = os.path.join(video_with_tracking_dir, \
                                     os.path.basename(video_path))
    # video_writer = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), 
                               fps, (height, width))
    


    pass    


def get_video_paths_from_dlc(dlc_processed_data, data_dir):
    video_dir = os.path.join(data_dir, 'video_files')
    video_times_and_paths = []
    for d in range(len(dlc_processed_data)):
        video_time = dlc_processed_data[d].columns[0][0]
        video_path = os.path.join(video_dir, "video_" + video_time + ".avi")
        video_times_and_paths.append((video_time, video_path))        
        
    return video_times_and_paths        


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(dlc_pickle_path)
    video_times_and_paths = get_video_paths_from_dlc(dlc_processed_data, data_dir)

    for d in enumerate(dlc_processed_data):
        
        video_time = d[1].columns[0][0]
        print(video_time)

        # find the correct video path for this video time
        for v in video_times_and_paths:
            if v[0] == video_time:
                video_path = v[1]
                break
        
        # create the video with the dlc data
        create_video_with_dlc_data(d[1], video_path)




    pass