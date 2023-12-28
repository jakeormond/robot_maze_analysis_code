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
                                     video_name)
    # video_writer = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path,
                               cv2.VideoWriter_fourcc('P', 'I', 'M', '1'), 
                               fps, (height, width))
    
    num_frames = dlc_data.shape[0]

    dlc_time = dlc_data.columns[0][0]
    for frame_idx in range(num_frames):
        ret, frame = cap.read()

        # get the dlc data for this frame
        dlc_for_frame = dlc_data.iloc[frame_idx, :]
       
        hd_for_frame = dlc_for_frame.loc[(dlc_time, 'hd')]

        if math.isnan(hd_for_frame):
            continue

        x1 = np.cos(hd_for_frame) * arrow_len
        y1 = -np.sin(hd_for_frame) * arrow_len

        x2 = np.cos(hd_for_frame + np.pi) * arrow_len
        y2 = -np.sin(hd_for_frame + np.pi) * arrow_len

        x_cropped = dlc_for_frame.loc[(dlc_time, 'x_cropped')]
        y_cropped = dlc_for_frame.loc[(dlc_time, 'y_cropped')]        
        
        arrow_start = (int(x_cropped + x1), 
                       int(y_cropped + y1))
        arrow_end = (int(x_cropped + x2), 
                     int(y_cropped + y2))
        
        cv2.arrowedLine(frame, arrow_end, arrow_start, 
                        (0, 0, 255), 4, tipLength = 0.5)

        cv2.startWindowThread()

        cv2.imshow('Video Player', frame)

        video_writer.write(frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break 
    
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()  


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

    for i, d in enumerate(dlc_processed_data):

        if i <= 1:
            continue
        
        video_time = d.columns[0][0]
        print(video_time)

        # find the correct video path for this video time
        for v in video_times_and_paths:
            if v[0] == video_time:
                video_path = v[1]
                break
        
        # create the video with the dlc data
        create_video_with_dlc_data(d, video_path)




    pass