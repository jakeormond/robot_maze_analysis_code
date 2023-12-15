# bonsai is not able to save files to the curent name generated
# by the robot maze program. Instead, it always saves the files with the 
# previously generated name. This script renames the bonsai files to the
# correct name.

import os
from get_directories import get_data_dir 
import glob
import time

def match_behaviour_and_bonsai_files(behaviour_dir, video_dir):  
    # find csv files in behaviour directory
    csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))
    # get the time that each csv file was created
    csv_times = [os.path.getmtime(f) for f in csv_files]
    csv_times = [time.localtime(t) for t in csv_times]
    # get the hour and minutes
    csv_times = [(t.tm_hour, t.tm_min) for t in csv_times]

    # find the video files in the video_files directory
    video_dir = os.path.join(data_dir, 'video_files')
    video_files = glob.glob(os.path.join(video_dir, '*.avi'))
    # get the time that each video file was created
    video_times = [os.path.getmtime(f) for f in video_files]
    video_times = [time.localtime(t) for t in video_times]
    video_times = [(t.tm_hour, t.tm_min) for t in video_times]

    print("csv times:" + str(csv_times))
    print("video times:" + str(video_times))
    
    # match each csv time to the corresponding video time
    bonsai_ind = [None] * len(csv_times)
    behaviour_and_matching_bonsai_files = [None] * len(csv_times)
    for c in range(len(csv_times)):
        csv_time = csv_times[c]
        for v in range(len(video_times)):
            video_time = video_times[v]
            if (csv_time == video_time) or \
                (csv_time[0] == video_time[0] and csv_time[1] == video_time[1] + 1) or \
                    (csv_time[0] == video_time[0] and csv_time[1] == video_time[1] - 1):
                # match the names
                bonsai_ind = v
                behaviour_and_matching_bonsai_files[c] = \
                    (csv_files[c], video_files[v])
                
                print(behaviour_and_matching_bonsai_files[c])                
                break

    return behaviour_and_matching_bonsai_files

def rename_bonsai_files(behaviour_and_matching_bonsai_files, video_dir, video_csv_dir):
    bonsai_times = [None] * len(behaviour_and_matching_bonsai_files)
    for i, b in enumerate(behaviour_and_matching_bonsai_files):
        # get the bonsai file name
        video_file = os.path.basename(b[1])
        bonsai_time = os.path.splitext(video_file)[0]
        # remove the video_ prefix
        bonsai_time = bonsai_time[6:]
        bonsai_times[i] = bonsai_time

    bonsai_files = ["cropTS", "cropValues", "dlcOut", "pulseTS", "videoTS"]
    for b in bonsai_files:
        # get list of all bonsai files starting with b
        bonsai_files_b = glob.glob(os.path.join(video_csv_dir, b + "*.csv"))
        
        # generate the list of files that should be there
        bonsai_files_b2 = [None] * len(bonsai_times)
        for i, t in enumerate(bonsai_times):
            bonsai_files_b2[i] = b + "_" + t + ".csv"

        # compare the two lists
        bonsai_files_b2 = set(bonsai_files_b2)
        bonsai_files_b = set(bonsai_files_b)
        # first check that all files in bonsai_files_b2 are in bonsai_files_b
        if not bonsai_files_b2.issubset(bonsai_files_b):
            raise Exception("Not all files in bonsai_files_b2 are in bonsai_files_b")
        
        # then move all files in bonsai_files_b not in bonsai_files_b2 to a new folder
        bonsai_files_b3 = bonsai_files_b - bonsai_files_b2
        bonsai_files_b3 = list(bonsai_files_b3)
        bonsai_files_b3_dir = os.path.join(video_csv_dir, "extra_bonsai_files")
        if not os.path.exists(bonsai_files_b3_dir):
            os.mkdir(bonsai_files_b3_dir)
        for b3 in bonsai_files_b3:
            b3_path = os.path.join(video_csv_dir, b3)
            b3_new_path = os.path.join(bonsai_files_b3_dir, b3)
            os.rename(b3_path, b3_new_path)
        
        # rename the bonsai files        
        
        bonsai_file = b + ".csv"
        bonsai_file_path = os.path.join(video_dir, bonsai_file)
        bonsai_file_new_path = os.path.join(video_csv_dir, bonsai_file)
        os.rename(bonsai_file_path, bonsai_file_new_path)



    pass



if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    video_dir = os.path.join(data_dir, 'video_files')
    video_csv_dir = os.path.join(data_dir, 'video_csv_files')
    behaviour_and_matching_bonsai_files = \
        match_behaviour_and_bonsai_files(behaviour_dir, video_dir)
    rename_bonsai_files(behaviour_and_matching_bonsai_files, \
                        video_dir, video_csv_dir)