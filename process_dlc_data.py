import os
import glob
import pandas as pd
import numpy as np
from numpy import matlib as mb
import re
import pickle
import pycircstat



def process_dlc_data(animal, session):
    # detect operating system
    if os.name == 'nt':
        home_dir = 'D:/analysis' # WINDOWS
    elif os.name == 'posix': # Linux or Mac OS
        home_dir = "/media/jake/LaCie" # Linux/Ubuntu
        
    data_dir = os.path.join(home_dir, animal, session)

    dlc_dir = os.path.join(data_dir, 'deeplabcut')

    # load tracking data
    tracking_files = glob.glob(os.path.join(dlc_dir, '*.h5'))
    tracking_files.sort()
    
    csv_directory = 'video_csv_files'
    csv_path = os.path.join(data_dir, csv_directory)

    dlc_processed_data = []

    body_col = 'spot1' # used to determine what platform the animal is on

    for file in tracking_files:    
        dlc = pd.read_hdf(file)
        column_names = dlc.columns.levels
        dlc = dlc[column_names[0].values[0]] # don't need top-level columnm, so get rid of it
        num_dlc_frames = dlc.shape[0]
        
        # GET THE TRIAL DATA AND TIME; these are needed to open the csv files that have 
        # video, pulse and crop timestamps and crop values
        end_ind = re.search('DLC',file)
        # get the string between "video_" and "DLC"
        slash_ind = file.find('video_') + 6
        trial_time = file[slash_ind:end_ind.span()[0]]
               
        ts_csv = 'videoTS_' + trial_time + '.csv'
        ts_csv = os.path.join(csv_path, ts_csv)
        video_ts = pd.read_csv(ts_csv, header=None)
        video_ts = video_ts[0]
        num_video_frames = video_ts.shape[0]
        assert num_dlc_frames == num_video_frames, 'unequal frame numbers'
        num_frames = num_dlc_frames
        del num_dlc_frames, num_video_frames
        
        pulse_csv = 'pulseTS_' + trial_time + '.csv'
        pulse_csv = os.path.join(csv_path, pulse_csv)
        pulse_ts = pd.read_csv(pulse_csv, header=None)
        pulse_ts = pulse_ts[0]

        crop_vals_csv = 'cropValues_' + trial_time + '.csv'
        crop_vals_csv = os.path.join(csv_path, crop_vals_csv)
        crop_vals = pd.read_csv(crop_vals_csv, header=None)
        crop_vals = crop_vals.rename(columns={0: 'x', 1: 'y', 2: 'height', 3: 'width'})

        crop_ts_csv = 'cropTS_' + trial_time + '.csv'
        crop_ts_csv = os.path.join(csv_path, crop_ts_csv)
        crop_ts = pd.read_csv(crop_ts_csv, header=None)
        crop_ts = crop_ts.rename(columns={0: 'ts'})

        crop_info = pd.concat([crop_ts, crop_vals], axis = 1)
        del crop_ts, crop_vals

        # HEAD TRACKING: identify frames where tracking hasn't been able to identify dots 1 and 4, which are 
        # the points used to track the animal and calculate head direction
        dot1_lklhood = dlc["dot1", "likelihood"]
        dot1_bad_frames = [i for i,v in enumerate(dot1_lklhood) if v < .9]
        
        dot4_lklhood = dlc["dot1", "likelihood"]
        dot4_bad_frames = [i for i,v in enumerate(dot4_lklhood) if v < .9]
        
        bad_frames = dot1_bad_frames + dot4_bad_frames
        bad_frames = list(set(bad_frames))
        bad_frames.sort()    
        consec_bad_frames = get_consec(bad_frames)

        # interpolate head positions with bad tracking
        dlc_interp = dlc[["dot1", "dot4"]]
        dlc_interp = dlc_interp.drop(('dot1', 'likelihood'), axis = 1)
        dlc_interp = dlc_interp.drop(('dot4', 'likelihood'), axis = 1)

        for idx, bad_frame_chunk in enumerate(consec_bad_frames):
            dlc_interp[bad_frame_chunk[0]:bad_frame_chunk[1]+1] = np.nan
        
        dlc_interp = dlc_interp.interpolate(limit_area='inside')
        
        # BODY TRACKING
        body_lklhood = dlc[body_col, "likelihood"]
        body_bad_frames = [i for i,v in enumerate(dot1_lklhood) if v < .9]
        consec_bad_body_frames = get_consec(body_bad_frames)
        
        dlc_body_interp = dlc[body_col]
        dlc_body_interp = dlc_body_interp.drop('likelihood', axis = 1)

        for idx, bad_frame_chunk in enumerate(consec_bad_body_frames):
            dlc_body_interp[bad_frame_chunk[0]:bad_frame_chunk[1]+1] = np.nan
        
        dlc_body_interp = dlc_body_interp.interpolate(limit_area='inside')
        
        # HEAD DIRECTION: calculate head direction
        x_diff = dlc_interp["dot1", "x"] - dlc_interp["dot4", "x"]
        y_diff = dlc_interp["dot4", "y"] - dlc_interp["dot1", "y"] # order reversed for y because why values start from top of screen, not bottom
        hd_rad = np.around(np.arctan2(y_diff, x_diff), 3)
        
        # CROPPING: correct x and y position for the effect of cropping
        crop_info.loc[0, 'ts'] = video_ts[0]
        crop_frames_temp = np.interp(crop_info['ts'], video_ts, list(range(num_frames)))
        crop_info['frame'] = np.ceil(crop_frames_temp)
        
        frames = crop_info['frame'].tolist()
        n_crops = len(crop_info)
        x_crops = np.zeros([num_frames, 1])
        y_crops = np.zeros([num_frames, 1])
        
        for i in range(n_crops): # create list of crop positions for each frame; this simplifies the correction
            x_crop = crop_info.loc[i, 'x']
            y_crop = crop_info.loc[i, 'y']
            first_frame = int(crop_info.loc[i, 'frame'])
        
            if i == n_crops-1:
                last_frame = int(num_frames-1)
            else:
                last_frame = int(crop_info.loc[i+1, 'frame'])
        
            n_reps = int(last_frame - first_frame + 1)    
        
            x_crops_temp = np.matlib.repmat(x_crop, n_reps, 1)    
            x_crops[first_frame:last_frame+1] = x_crops_temp
            
            y_crops_temp = np.matlib.repmat(y_crop, n_reps, 1)    
            y_crops[first_frame:last_frame+1] = y_crops_temp
        
        x_crops = x_crops.astype(int) # these need to be integers 
        y_crops = y_crops.astype(int)
        
        # CREATE DATAFRAME WITH CROPPING CORRECTION
        # create a dataframe of positional information corrected for cropping 
        x1_uncor = dlc_interp['dot1', 'x'].to_numpy()
        y1_uncor = dlc_interp['dot1', 'y'].to_numpy()
        x2_uncor = dlc_interp['dot4', 'x'].to_numpy()
        y2_uncor = dlc_interp['dot4', 'y'].to_numpy()
        x_body_uncor = dlc_body_interp['x'].to_numpy()
        y_body_uncor = dlc_body_interp['y'].to_numpy()  
        
        x1_cor = np.add(np.squeeze(x_crops), x1_uncor)
        y1_cor = np.add(np.squeeze(y_crops), y1_uncor)
        x2_cor = np.add(np.squeeze(x_crops), x2_uncor)
        y2_cor = np.add(np.squeeze(y_crops), y2_uncor)
        x_body_cor = np.add(np.squeeze(x_crops), x_body_uncor)
        y_body_cor = np.add(np.squeeze(y_crops), y_body_uncor)

        x_uncor = np.mean(np.concatenate((x1_uncor.reshape(-1,1), 
                    x2_uncor.reshape(-1,1)), axis=1), axis=1) 
        x_uncor = np.around(x_uncor, 3)
        y_uncor = np.mean(np.concatenate((y1_uncor.reshape(-1,1), 
                    y2_uncor.reshape(-1,1)), axis=1), axis=1)
        y_uncor = np.around(y_uncor, 3)
        
        x_cor = np.mean(np.concatenate((x1_cor.reshape(-1,1), 
                    x2_cor.reshape(-1,1)), axis=1), axis=1)    
        x_cor = np.around(x_cor, 3)
        y_cor = np.mean(np.concatenate((y1_cor.reshape(-1,1), 
                    y2_cor.reshape(-1,1)), axis=1), axis=1)
        y_cor = np.around(y_cor, 3)
        
        x_body_uncor = np.around(x_body_uncor, 3)
        y_body_uncor = np.around(y_body_uncor, 3)
        x_body_cor = np.around(x_body_cor, 3)
        y_body_cor = np.around(y_body_cor, 3)    
        
        df_data = {'ts': video_ts, 'x': x_cor, 'y': y_cor, 
                'x_cropped': x_uncor, 'y_cropped': y_uncor, 
                'x_body': x_body_cor, 'y_body': y_body_cor, 
                'x_body_cropped': x_body_uncor, 'y_body_cropped': y_body_uncor,
                'hd': hd_rad}
        df = pd.DataFrame(data=df_data)
        df.columns = pd.MultiIndex.from_product([[trial_time], df.columns])
        
        dlc_processed_data.append(df)
    
    return dlc_processed_data
        
        

# a function to identify consecutive number, used below to get 
# continuous chunks of bad tracking for interpolation
def get_consec(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + [item for sublist in gaps for item in sublist] + nums[-1:])
    return list(zip(edges, edges))

if __name__ == "__main__":
    animal = 'Rat64'
    session = '08-11-2023'
    dlc_processed_data = process_dlc_data(animal, session)
    pass
