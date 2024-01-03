import os
import glob
import pandas as pd
import numpy as np
import numpy.matlib as mb
import re
import pickle
import copy
from get_directories import get_data_dir
from get_pulses import load_bonsai_pulses

# note that sample rate is hardcoded as 30000
# create global variable for sample rate
sample_rate = 30000

def process_dlc_data(dlc_dir):  
    # load tracking data
    tracking_files = glob.glob(os.path.join(dlc_dir, '*.h5'))
    tracking_files.sort()
    
    csv_directory = 'video_csv_files'
    csv_path = os.path.join(data_dir, csv_directory)

    dlc_processed_data = {}

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
        
            x_crops_temp = mb.repmat(x_crop, n_reps, 1)    
            x_crops[first_frame:last_frame+1] = x_crops_temp
            
            y_crops_temp = mb.repmat(y_crop, n_reps, 1)    
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
        # df.columns = pd.MultiIndex.from_product([[trial_time], df.columns])
        
        dlc_processed_data[trial_time] = df
    

    # save the processed data to a pickle file
    pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(dlc_processed_data, f)
    
    return dlc_processed_data, pickle_path


def get_video_times_in_samples(dlc_processed_data, pulses):
    """
    Get the video times in samples. This is done by interpolating the video 
    timestamps with the bonsai timestamps. The last few frames after the last 
    pulse can't be interpolated properly so they need to be manually calculated.
    
    Parameters
    ----------
    dlc_processed_data : list of pandas dataframes
        The dlc processed data. Each element in the list is a trial. Each 
        dataframe contains the dlc processed data for a trial. Contains only
        video times in ms, not in samples, which are necessary to calculate
        the times of the recorded spikes. 
    pulses : list of pandas dataframes
        The pulse times and samples. Each element in the list is a trial. 
        
    Returns
    -------
    dlc_processed_with_samples : list of pandas dataframes
        The dlc processed data with the video times in samples. Each element in 
        the list is a trial. Each dataframe contains the dlc processed data for 
        a trial. The first column is the video time in ms and the second column 
        is the video time in samples. The rest of the columns are the body 
        part positions and head direction.
    """
    trial_times = list(dlc_processed_data.keys())
    dlc_processed_with_samples = {}

    for i, t in enumerate(trial_times):
       
        d = dlc_processed_data[t]
        video_ts = d['ts']
        pulses_ts = pulses[t]['bonsai_pulses_ms']
        pulses_samples = pulses[t]['imec_pulses_samples']

        # interpolate the video samples 
        video_samples = np.interp(video_ts, pulses_ts, pulses_samples)
        video_samples = np.round(video_samples).astype(int)

        # the last few frames after the last pulse can't be interpolated properly
        # so they need to be manually calculated
        # first find the first video_ts that is greater than the last pulse_ts
        last_pulse_ts = pulses_ts.iloc[-1]
        last_video_ind = np.where(video_ts > last_pulse_ts)[0][0]
        video_samples[last_video_ind:] = video_samples[last_video_ind - 1] + \
            (video_ts[last_video_ind:] - video_ts[last_video_ind - 1])*(sample_rate/1000) # sample rate 30000 divide by 1000 (i.e. 30 samples per ms)

        # add the video samples to the dlc_processed_data.
        # make it the second column
        dlc_processed_with_samples[t] = copy.deepcopy(d)
        # insert the video samples as the second column
        dlc_processed_with_samples[t].insert(1, 'video_samples', video_samples)
        
    return dlc_processed_with_samples


def restrict_dlc_to_video_endpoints(dlc_processed_data, video_endpoints):
    
    trial_times = list(dlc_processed_data.keys())
    
    restricted_dlc_processed_data = {}
    for t in trial_times:
        dlc_temp = dlc_processed_data[t].loc[0:video_endpoints[t]['end_frame']]
        restricted_dlc_processed_data[t] = dlc_temp
    
    return restricted_dlc_processed_data



def load_dlc_processed_pickle(pickle_path):        
    with open(pickle_path, 'rb') as f:
        dlc_processed_data = pickle.load(f)
    
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
    data_dir = get_data_dir(animal, session)
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    # dlc_processed_data, pickle_path = process_dlc_data(dlc_dir)
    # del dlc_processed_data
    pickle_path = os.path.join(dlc_dir, 'dlc_processed_data.pkl')
    dlc_processed_data = load_dlc_processed_pickle(pickle_path)
    
    # load the pulses, which contains both the bonsai and spikeglx pulses in 
    # ms and samples, respectively
    bonsai_dir = os.path.join(data_dir, 'video_csv_files')
    pulses = load_bonsai_pulses(bonsai_dir)

    dlc_processed_with_samples = get_video_times_in_samples(dlc_processed_data, pulses)
       
    # save the processed data to a pickle file
    pickle_path = os.path.join(dlc_dir, 'dlc_processed_data_with_samples.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(dlc_processed_with_samples, f)

    # delete dlc_processed_with samples
    del dlc_processed_with_samples

    # load the processed data with samples
    with open(pickle_path, 'rb') as f:
        dlc_processed_with_samples = pickle.load(f)
    

    video_dir = os.path.join(data_dir, 'video_files')
    video_endpoints_file = os.path.join(video_dir, 'video_endpoints.pkl')
    # load the pickle file
    with open(video_endpoints_file, 'rb') as f:
        video_endpoints = pickle.load(f)

    dlc_processed_data = restrict_dlc_to_video_endpoints(dlc_processed_data, 
                                                         video_endpoints)

    pass





