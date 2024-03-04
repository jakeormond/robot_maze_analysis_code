# bonsai is not able to save files to the curent name generated
# by the robot maze program. Instead, it always saves the files with the 
# previously generated name. This script renames the bonsai files to the
# correct name.

import os
from get_directories import get_data_dir 
import glob
import time


def match_behaviour_and_bonsai_datestamps(behaviour_dir, video_dir):  
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
    behaviour_and_matching_bonsai_datestamps = [None] * len(csv_times)
    for c in range(len(csv_times)):
        csv_time = csv_times[c]
        for v in range(len(video_times)):
            video_time = video_times[v]
            if (csv_time == video_time) or \
                (csv_time[0] == video_time[0] and csv_time[1] == video_time[1] + 1) or \
                    (csv_time[0] == video_time[0] and csv_time[1] == video_time[1] - 1):
                # match the names
                bonsai_ind = v
                csv_file = csv_files[c]
                csv_date = os.path.basename(csv_file)
                # remove the .csv
                csv_date = csv_date[:-4]

                video_file = video_files[v]
                video_date = os.path.basename(video_file)
                # remove the "video_" prefix and the .avi
                video_date = video_date[6:-4]

                behaviour_and_matching_bonsai_datestamps[c] = \
                    (csv_date, video_date)
                
                print(behaviour_and_matching_bonsai_datestamps[c])                
                break

    return behaviour_and_matching_bonsai_datestamps


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


def re_date_files(behaviour_and_matching_files, file_paths):
    # loop backwards through behaviour_and_matching_bonsai_files
    for i, b in reversed(list(enumerate(behaviour_and_matching_files))):
        wrong_date = b[1]
        correct_date = b[0]

        # find file path with wrong date
        for f in file_paths:
            if wrong_date in f:
                # rename the file
                f_new = f.replace(wrong_date, correct_date)
                os.rename(f, f_new)
                break


def move_extra_files(behaviour_and_matching_times, files, directory, subdirectory):
    for f in files:
        # get datestamp of file
        file_name = os.path.basename(f)
        file_name = os.path.splitext(file_name)[0]
        # find where date stats with "20"
        date_ind = file_name.find("20")
        file_date = file_name[date_ind:date_ind+19]

        match_flag = False
        for b in behaviour_and_matching_times:
            behaviour_time = b[1]
            if file_date == behaviour_time:
                match_flag = True
                break

        if not match_flag:            
            # check if subdirectory exists
            if not os.path.exists(os.path.join(directory, subdirectory)):
                os.mkdir(os.path.join(directory, subdirectory))
            
            # move the file to subdirectory
            f_new_path = os.path.join(directory, subdirectory, os.path.basename(f))
            os.rename(f, f_new_path) 


def re_date_bonsai_files(behaviour_and_matching_bonsai_datestamps, video_csv_dir):
    bonsai_files = ["cropTS", "cropValues", "dlcOut", "pulseTS", "videoTS"]
    for b in bonsai_files:
        # find all bonsai files starting with b
        bonsai_files_b = glob.glob(os.path.join(video_csv_dir, b + "*.csv"))

        # move any that don't have times from matching_timestamps to a new folder
        move_extra_files(behaviour_and_matching_bonsai_datestamps, bonsai_files_b, \
                         video_csv_dir, "extra_bonsai_files")
        
        # rename the bonsai files
        bonsai_files_b = glob.glob(os.path.join(video_csv_dir, b + "*.csv"))
        re_date_files(behaviour_and_matching_bonsai_datestamps, bonsai_files_b)

    
def re_date_video_files(behaviour_and_matching_video_datestamps, video_dir):
    video_files = glob.glob(os.path.join(video_dir, "*.avi"))

    # move any that don't have times from matching_timestamps to a new folder
    move_extra_files(behaviour_and_matching_video_datestamps, video_files, \
                         video_dir, "extra_video_files")

    # rename the video files
    video_files = glob.glob(os.path.join(video_dir, "*.avi"))
    re_date_files(behaviour_and_matching_video_datestamps, video_files)


def revert_dlc_file_names(dlc_dir):
    # dlc doesn't like our file names, since we have periods between hours, minutes, and seconds
    # so we change these to hyphes. Now, we change the hyphens back to periods for consistency 
    # with our other files
    dlc_file_types = [".h5", ".pickle"]
    for t in dlc_file_types:
        dlc_files = glob.glob(os.path.join(dlc_dir, "*" + t))

        for d in dlc_files:
            # get the file name
            d_name = os.path.basename(d)
            # find indices of third and fourth hyphen
            hyphen_inds = [i for i in range(len(d_name)) if d_name[i] == "-"]
            
            # replace the third and fourth hyphen with periods
            for i in range(2, 4):
                d_new = d_name[:hyphen_inds[i]] + '.' + d_name[hyphen_inds[i]+1:]
                d_name = d_new

            # add the full path back to d_new
            d_new_path = os.path.join(dlc_dir, d_name)

            os.rename(d, d_new_path)
           

def re_date_dlc_files(behaviour_and_matching_video_datestamps, dlc_dir):
    dlc_file_types = [".h5", ".pickle"]
    for t in dlc_file_types:
        dlc_files = glob.glob(os.path.join(dlc_dir, "*" + t))

        # move any that don't have times from matching_timestamps to a new folder
        move_extra_files(behaviour_and_matching_video_datestamps, dlc_files, \
                            dlc_dir, "extra_dlc_files")

        # rename the video files
        dlc_files = glob.glob(os.path.join(dlc_dir, "*" + t))
        re_date_files(behaviour_and_matching_video_datestamps, dlc_files)
    

if __name__ == "__main__":
    animal = 'Rat46'
    session = '20-02-2024'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    video_dir = os.path.join(data_dir, 'video_files')
    video_csv_dir = os.path.join(data_dir, 'video_csv_files')
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    behaviour_and_matching_bonsai_datestamps = \
        match_behaviour_and_bonsai_datestamps(behaviour_dir, video_dir)
    
    # redate bonsai files
    # re_date_bonsai_files(behaviour_and_matching_bonsai_datestamps, \
    #                     video_csv_dir)
        
    
    revert_dlc_file_names(dlc_dir)

    # redate dlc files
    re_date_dlc_files(behaviour_and_matching_bonsai_datestamps, \
                        dlc_dir)    
    
   # redate video files
    re_date_video_files(behaviour_and_matching_bonsai_datestamps, \
                        video_dir) 
