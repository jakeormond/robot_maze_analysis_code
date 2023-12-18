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
    behaviour_times = [None] * len(behaviour_and_matching_bonsai_files)
    for i, b in enumerate(behaviour_and_matching_bonsai_files):
        # get the bonsai file name
        video_file = os.path.basename(b[1])
        bonsai_time = os.path.splitext(video_file)[0]
        # remove the video_ prefix
        bonsai_time = bonsai_time[6:]
        bonsai_times[i] = bonsai_time

        # get the behaviour file name
        behaviour_file = os.path.basename(b[0])
        behaviour_time = os.path.splitext(behaviour_file)[0]
        behaviour_times[i] = behaviour_time


    bonsai_files = ["cropTS", "cropValues", "dlcOut", "pulseTS", "videoTS"]
    for b in bonsai_files:
        # get list of all bonsai files starting with b
        bonsai_files_b = glob.glob(os.path.join(video_csv_dir, b + "*.csv"))
        
        # generate the list of files that should be there
        bonsai_files_b2 = [None] * len(bonsai_times)
        for i, t in enumerate(bonsai_times):
            bonsai_files_b2[i] = os.path.join(video_csv_dir, b + "_" + t + ".csv")

        # compare the two lists
        bonsai_files_b2_set = set(bonsai_files_b2)
        bonsai_files_b_set = set(bonsai_files_b)
        # first check that all files in bonsai_files_b2 are in bonsai_files_b
        if not bonsai_files_b2_set.issubset(bonsai_files_b_set):
            raise Exception("Not all files in bonsai_files_b2 are in bonsai_files_b")
        
        # then move all files in bonsai_files_b not in bonsai_files_b2 to a new folder
        bonsai_files_b3 = bonsai_files_b_set - bonsai_files_b2_set
        bonsai_files_b3 = list(bonsai_files_b3)
        # check if bonsai_files_b3 is empty
        if len(bonsai_files_b3) != 0:           
            bonsai_files_b3_dir = os.path.join(video_csv_dir, "extra_bonsai_files")
            if not os.path.exists(bonsai_files_b3_dir):
                os.mkdir(bonsai_files_b3_dir)
            for b3 in bonsai_files_b3:
                b3_path = os.path.join(video_csv_dir, b3)
                b3_new_path = os.path.join(bonsai_files_b3_dir, b3)
                os.rename(b3_path, b3_new_path)
        
        # rename the bonsai files   
        # for i, b2 in reversed(list(enumerate(bonsai_files_b))):           
        #     # find where the date starts in b, the "20" that comes after b
        #     date_ind_b = b2.find(b) + len(b) + 1           
        #     bonsai_file = b2[:date_ind_b] + behaviour_times[i] + ".csv"              
        #     bonsai_file_new_path = os.path.join(video_csv_dir, bonsai_file)
        #     os.rename(b2, bonsai_file_new_path)
        
    # video files
    video_files = glob.glob(os.path.join(video_dir, "*.avi"))
    
    # generate the list of files that should be there
    video_files_b = [None] * len(bonsai_times)
    for i, t in enumerate(bonsai_times):
        video_files_b[i] = os.path.join(video_dir, "video_" + t + ".avi")

    # compare the two lists
    video_files_set = set(video_files)
    video_files_b_set = set(video_files_b)
    # first check that all files in bonsai_files_b2 are in bonsai_files_b
    if not video_files_b_set.issubset(video_files_set):
        raise Exception("Not all files in video_files_b are in video_files")
    
    # then move all files in bonsai_files_b not in bonsai_files_b2 to a new folder
    video_files_b2 = video_files_b_set - video_files_set
    video_files_b2 = list(video_files_b2)
    # check if video_files_b2 is empty
    if len(video_files_b2) != 0:           
        video_files_b2_dir = os.path.join(video_dir, "extra_video_files")
        if not os.path.exists(video_files_b2_dir):
            os.mkdir(video_files_b2_dir)
        for b2 in video_files_b2:
            b2_path = os.path.join(video_dir, b2)
            b2_new_path = os.path.join(video_files_b2_dir, b2)
            os.rename(b2_path, b2_new_path)
        
        # # rename the video files   
        # for i, v in reversed(list(enumerate(video_files))):           
        #     # find where the date starts in b, the "20" that comes after b
        #     date_ind = v.find("video_2") + 6      
        #     video_file = v[:date_ind]    
        #     video_file = video_file + behaviour_times[i] + ".avi"              
        #     os.rename(v, video_file)

    # dlc files
    dlc_files = [".h5", ".pickle"]
    for d in dlc_files:
        # get list of all dlc files end with d
        dlc_files_d = glob.glob(os.path.join(dlc_dir, "video" + d))

        # get generic dlc_file name with date and time removed
        dlc_file = os.path.basename(dlc_files_d[0])
        dlc_ind = dlc_file.find("DLC")
        dlc_file_end = dlc_file[dlc_ind:]

        # generate the list of files that should be there
        dlc_files_d2 = [None] * len(bonsai_times)
        for i, t in enumerate(bonsai_times):
            dlc_files_d2[i] = os.path.join(dlc_dir, "video_" + t + dlc_file_end + d)

        # compare the two lists
        dlc_files_d2_set = set(dlc_files_d2)
        dlc_files_d_set = set(dlc_files_d)
        # first check that all files in dlc_files_d2 are in dlc_files_d
        if not dlc_files_d2.issubset(dlc_files_d_set):
            raise Exception("Not all files in dlc_files_d2 are in dlc_files_d")
        
        # then move all files in bonsai_files_b not in bonsai_files_b2 to a new folder
        dlc_files_d3 = dlc_files_d_set - dlc_files_d2_set
        dlc_files_d3 = list(dlc_files_d3)
        # check if dlc_files_d3 is empty
        if len(dlc_files_d3) != 0:           
            dlc_files_d3_dir = os.path.join(dlc_dir, "extra_dlc_files")
            if not os.path.exists(dlc_files_d3_dir):
                os.mkdir(dlc_files_d3_dir)
            for d3 in dlc_files_d3:
                d3_path = os.path.join(dlc_dir, d3)
                d3_new_path = os.path.join(dlc_files_d3_dir, d3)
                os.rename(d3_path, d3_new_path)
        
        # rename the dlc files   
        for i, d2 in reversed(list(enumerate(dlc_files_d))):           
            # find where the date starts in d2, the "20" that comes after b
            date_ind_d = d2.find("video_2") + 6          
            dlc_file_new = "video_" + behaviour_times[i] + dlc_file_end + d              
            dlc_file_new_path = os.path.join(dlc_dir, dlc_file_new)
            os.rename(d2, dlc_file_new_path)
    pass



if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    video_dir = os.path.join(data_dir, 'video_files')
    video_csv_dir = os.path.join(data_dir, 'video_csv_files')
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    behaviour_and_matching_bonsai_files = \
        match_behaviour_and_bonsai_files(behaviour_dir, video_dir)
    rename_bonsai_files(behaviour_and_matching_bonsai_files, \
                        video_dir, video_csv_dir, dlc_dir)
    
