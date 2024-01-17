
# rename_bonsai_files.py		

Fixes the datestamp issue with Bonsai, where it is always one 
filename behind the honeycomb task script. Uses the times when 
the files were modified to correctly match them. 
RUN THIS BEFORE RUNNING ANY OTHER CODE!!!!!

# extract_pulses_from_raw.py	

Finds the digital pulses emitted by Bonsai and recorded by SpikeGLX 
by reading the final channel of the binary file. 

# get_pulses.py			

Reads in the pulse timings recorded by Bonsai, and matches them to the 
pulses read from the SpikeGLX data. It produces plots to allow the 
user to confirm that the SpikeGLX pulses are correctly aligned to the 
Bonsai data. The Bonsai pulses (in computer time (ms)) and the SpikeGLX 
pulses (in samples) are combined into a 2d array and saved as a list of 
arrays as an H5 file. Figures are plotted to allow the user to confirm 
that pulses have been correctly aligned (see below).

![pulse alignment](media/2023-11-10_13.31.00.png)


# process_dlc_data.py		

Extracts the positional data from deeplabcut
NOTE THAT POSITIONAL DATA NEEDS TO BE TRIMMED AT START AND END!!!
THIS CAN BE DONE WHEN PROCESSING NEURAL DATA. 

ALSO NOTE THAT: process_dlc_data.py now contains the functionality 
of calculate_video_times.py, which can now be removed. 
				
# get_video_endpoints		

Video recording, in most cases, went beyond the end of the trial (i.e. 
the experimenter can be seen at the end of the videos coming in to the 
arena with the food reward), and so an endpoint for each video needs 
to be determined. The user needs to open the video in VLC, scan to what
they determine to be the end of the video, and enter the end time in 
minutes:seconds format (as can be viewed directly from VLC).

# create_videos_with_dlc_data.py	

Produces videos with a red arrow tracking head position and direction. 
In the cropped videos it produces, it super-imposes a red END over the 
frames that come past the endpoint for manual verification. 

![cropped video](media/video_2023-11-08_16.52.26.gif)



# get_pulses.py			

Reads in the pulse timings recorded by Bonsai, and matches them to the 
pulses read from the SpikeGLX data. It produces plots to allow the 
user to confirm that the SpikeGLX pulses are correctly aligned to the 
Bonsai data. The Bonsai pulses (in computer time (ms)) and the SpikeGLX 
pulses (in samples) are combined into a 2d array and saved as a list of 
arrays as an H5 file.
				
# calculate_video_times.py	

Reads in the pulses saved in get_pulses.py, and the dlc_processed_data 
in process_dlc_data.py. dlc_processed_data contains the positional 
information for each video frame, as well as its timestamp in cpu time. 
This script then interpolates the timestamps in sample time so that the
positional data can be aligned with the neural data. The data is saved
in a pickle file called called dlc_processed_data_with_sample.pkl.

# load_behaviour.py		

Loads in the behavioural csv files, and saves the data as a list of 
dataframes with the name property corresponding to the trial time, 
and the goal property corresponding to the goal on that trial (saved as 
behaviour_data.pkl). Also, sorts trials according to the goal, and 
saves them as a dictionary (with the goal as key) of lists corresponding 
to each goal (behaviour_data_by_goal.pkl).
Note: this contains code showing how to save and load dataframes with custom
attributes. Unfortunately, just straight up saving them doesn't work, so 
the attributes need to be stripped out into their own lists, stored within
a dictionary, and then added back to the dataframes at loading time. 

# calculate_pos_and_dir.py

Calculates the animal's current platform, as well as the goal directions from 
its current position, and its head direction relative to each of the goals.

# load_sorted_spikes.py		

Loads the clusters classified as good in Phy (need to verify the cluster 
qualities are mine, and not kilosort's). Saves them in a dictionary with the 
cluster numbers as the keys, and the spike times in samples as the data. 
File is unit_spike_times.pickle. 

# classify_neurons.py 		

Calculates mean firing rates and average waveforms, from which spike width 
can be calculated. Plots a scatter plot of mean rates vs spike width, 
from which it is possible to manually identify the interneurons. Once we 
have more sessions to look at, we can determine thresholds/cutoffs to 
automate the identification of principal and interneurons; for now, go 
through the plots, and manually enter a spike-width cut-off. 
				
