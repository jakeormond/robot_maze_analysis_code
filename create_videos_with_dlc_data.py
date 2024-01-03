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


def create_cropped_video_with_dlc_data(dlc_data, video_path, video_endpoint):
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
    end_frame = video_endpoint['end_frame']

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
        
        if frame_idx > end_frame:
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

def create_full_video_with_dlc_data(dlc_data, video_path, video_endpoint):
    # blank full size frame 
    blank_fs_frame = np.zeros((2048, 2448))

    arrow_len = 10
    cap = cv2.VideoCapture(video_path)

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
                               fps, (width, height), isColor=False)
    
    num_frames = dlc_data.shape[0]
    end_frame = video_endpoint['end_frame']

    dlc_time = dlc_data.columns[0][0]
    for frame_idx in range(num_frames):
        ret, frame = cap.read()

        if frame_idx > end_frame:
            break

        if frame_idx % 10 != 0:
            continue

        # get the dlc data for this frame
        dlc_for_frame = dlc_data.iloc[frame_idx, :]
        hd_for_frame = dlc_for_frame.loc[(dlc_time, 'hd')]

        if math.isnan(hd_for_frame):
            continue

        x_crop_pos = int(dlc_for_frame[(dlc_time, 'x')] 
                         - dlc_for_frame[(dlc_time, 'x_cropped')])
        y_crop_pos = int(dlc_for_frame[(dlc_time, 'y')] 
                         - dlc_for_frame[(dlc_time, 'y_cropped')])
       
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fs_frame = blank_fs_frame
        fs_frame[y_crop_pos:y_crop_pos + 600, x_crop_pos:x_crop_pos + 600] = gray_frame
        fs_frame = fs_frame.astype('uint8')        

        x1 = np.cos(hd_for_frame) * arrow_len
        y1 = -np.sin(hd_for_frame) * arrow_len

        x2 = np.cos(hd_for_frame + np.pi) * arrow_len
        y2 = -np.sin(hd_for_frame + np.pi) * arrow_len

        x = dlc_for_frame.loc[(dlc_time, 'x')]
        y = dlc_for_frame.loc[(dlc_time, 'y')]        
        
        arrow_start = (int(x + x1), 
                       int(y + y1))
        arrow_end = (int(x + x2), 
                     int(y + y2))
        
        cv2.arrowedLine(frame, arrow_end, arrow_start, 
                        (0, 0, 255), 4, tipLength = 0.5)
        
        # shrink the frame to keep file size under control
        fs_frame_re = cv2.resize(fs_frame, (1224, 1024))

        cv2.startWindowThread()

        cv2.imshow('Video Player', frame)

        video_writer.write(fs_frame_re)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break 
        
    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()

def get_video_paths_from_dlc(dlc_processed_data, data_dir):
    video_dir = os.path.join(data_dir, 'video_files')
    video_paths = {}
    for d in range(len(dlc_processed_data)):
        video_time = dlc_processed_data[d].columns[0][0]
        video_path = os.path.join(video_dir, "video_" + video_time + ".avi")
        video_paths[video_time] = video_path        
        
    return video_paths        


if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    dlc_pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(dlc_pickle_path)
    video_paths = get_video_paths_from_dlc(dlc_processed_data, data_dir)

    video_dir = os.path.join(data_dir, 'video_files')
   
    # load video_endpoints.pkl
    video_endpoints_path = os.path.join(video_dir, 'video_endpoints.pkl')
    with open(video_endpoints_path, 'rb') as f:
        video_endpoints = pickle.load(f)

    for i, d in enumerate(dlc_processed_data):

        video_time = d.columns[0][0]
        print(video_time)

        # find the correct video path for this video time
        video_path = video_paths[video_time]
        
        # get video endpoint
        video_endpoint = video_endpoints[video_time]

        # create thes video with the dlc data
        if i != 0:
            create_cropped_video_with_dlc_data(d, video_path, video_endpoint)
            
        create_full_video_with_dlc_data(d, video_path, video_endpoint)




    pass