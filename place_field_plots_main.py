import os
import glob
import shutil
import numpy as np

from get_directories import get_data_dir, get_robot_maze_directory
from rename_bonsai_files import match_behaviour_and_bonsai_datestamps, re_date_bonsai_files, re_date_video_files, re_date_dlc_files
from load_and_save_data import load_pickle, save_pickle
from get_pulses import load_imec_pulses, make_dataframe_pulse_numbers, match_bonsai_and_imec_pulses, plot_pulse_alignment
from get_video_endpoints import get_video_endpoints, get_video_startpoints
from process_dlc_data import process_dlc_data, get_video_times_in_samples, restrict_dlc_to_video_start_and_end, interpolate_out_nans
from load_behaviour import get_behaviour_dir, load_behaviour_file, split_behaviour_data_by_goal, split_dictionary_by_goal
from calculate_pos_and_dir import get_screen_coordinates, get_uncropped_platform_coordinates, get_current_platform, get_relative_head_direction, get_distances
from load_sorted_spikes import load_sorted_spikes
from restrict_spikes_to_trials import restrict_spikes_to_trials
from calculate_occupancy import get_goal_coordinates, calculate_frame_durations, concatenate_dlc_data, get_axes_limits, get_positional_occupancy, plot_occupancy_heatmap, get_directional_occupancy_from_dlc, plot_directional_occupancy, get_directional_occupancy_by_position, concatenate_dlc_data_by_goal
from calculate_spike_pos_hd import get_unit_position_and_directions, bin_spikes_by_position, smooth_rate_maps, bin_spikes_by_direction, bin_spikes_by_position_and_direction, check_bad_vals, sort_units_by_goal
from plot_spikes_and_pos import get_x_and_y_limits, plot_spikes_and_pos, plot_spikes_2goals, plot_spike_rates_by_direction, plot_rate_maps, plot_rate_maps_2goals, plot_spike_rates_by_direction_2goals


if __name__ == "__main__":
    animal = 'Rat47'
    session = '08-02-2024'
    data_dir = get_data_dir(animal, session)
    behaviour_dir = os.path.join(data_dir, 'behaviour')
    video_dir = os.path.join(data_dir, 'video_files')
    video_csv_dir = os.path.join(data_dir, 'video_csv_files')
    dlc_dir = os.path.join(data_dir, 'deeplabcut')

    starting_code_block = 3

    ################ CODE BLOCK 0 ##################
    ################ REDATE FILES ##################
    if starting_code_block == 0:
        behaviour_and_matching_bonsai_datestamps = \
            match_behaviour_and_bonsai_datestamps(behaviour_dir, video_dir)
        
        # print(behaviour_and_matching_bonsai_datestamps)
        # pause execution to allow user to check the re-dated files
        print('Check the re-dated files in the video directory and the behaviour directory.')
        input('Press enter to continue...')
        
        # redate bonsai files
        re_date_bonsai_files(behaviour_and_matching_bonsai_datestamps, \
                            video_csv_dir)
        
        # redate dlc files
        re_date_dlc_files(behaviour_and_matching_bonsai_datestamps, \
                            dlc_dir)    
        
    # redate video files
        re_date_video_files(behaviour_and_matching_bonsai_datestamps, \
                            video_dir) 


    ################# EXTRACT PULSES FROM BINARY ##################
    # do this in a separate script


    ################# CODE BLOCK 1 ##################
    ################## MATCH THE IMEC AND BONSAI PULSES ##################
    if starting_code_block <= 1:
        imec_pulses = load_imec_pulses(os.path.join(data_dir, 'imec_files'))
        n_pulses_and_samples = load_pickle('n_pulses_and_samples.pkl', os.path.join(data_dir, 'imec_files'))

        pulses_df = make_dataframe_pulse_numbers(n_pulses_and_samples, data_dir)
        # print the entire dataframe for visual inspection
        print(pulses_df)

        pulses = match_bonsai_and_imec_pulses(n_pulses_and_samples, data_dir)

        plot_pulse_alignment(pulses, data_dir)

        # ask user to check the alignment of the pulses
        print('Check the pulse alignment plots.')
        input('Press enter to continue...')


    ################# CODE BLOCK 2 ##################
    ################## GET VIDEO ENDPOINTS #############################
    if starting_code_block <= 2:
        video_dir = os.path.join(data_dir, 'video_files')

        endpoints = get_video_endpoints(video_dir, user_input=False)
        save_pickle(endpoints, 'video_endpoints', video_dir)


    ################# CODE BLOCK 3 ##################
    ################## PROCESS DEEPLABCUT DATA #############################
    if starting_code_block <= 3:
        dlc_dir = os.path.join(data_dir, 'deeplabcut')

        # process the dlc data
        dlc_processed_data = process_dlc_data(data_dir)
        save_pickle(dlc_processed_data, 'dlc_processed_data', dlc_dir)
        # dlc_processed_data = load_pickle('dlc_processed_data', dlc_dir) 

        # get the video startpoints
        video_dir = os.path.join(data_dir, 'video_files')
        video_startpoints = get_video_startpoints(dlc_processed_data)
        save_pickle(video_startpoints, 'video_startpoints', video_dir)

        # load the pulses, which contains both the bonsai and spikeglx pulses in 
        # ms and samples, respectively
        # pulses = load_bonsai_pulses(data_dir)
        video_csv_dir = os.path.join(data_dir, 'video_csv_files')
        pulses = load_pickle('pulses_dataframes', video_csv_dir)

        dlc_processed_with_samples = get_video_times_in_samples(dlc_processed_data, pulses)
        save_pickle(dlc_processed_with_samples, 'dlc_processed_with_samples', dlc_dir)
        
        # Once we have aligned the video data with the pulses recorded by the imec system, 
        # we can restrict the video data to the start and end of the video. 
        video_dir = os.path.join(data_dir, 'video_files')
        # video_startpoints = load_pickle('video_startpoints', video_dir)
        video_endpoints = load_pickle('video_endpoints', video_dir)

        dlc_final = restrict_dlc_to_video_start_and_end(dlc_processed_with_samples, 
                                            video_startpoints, video_endpoints)
        
        # identify any NaN in the data, and if found, interpolate them out,
        # or chop them off if they are at the beginning or end of the data
        dlc_final, save_flag = interpolate_out_nans(dlc_final)
        save_pickle(dlc_final, 'dlc_final', dlc_dir)


    ################### CODE BLOCK 4 ##################
    #################### PROCESS BEHAVIOUR DATA ############################
    if starting_code_block <= 4:
        behaviour_dir = get_behaviour_dir(data_dir)

        # find csv files in behaviour directory
        csv_files = glob.glob(os.path.join(behaviour_dir, '*.csv'))

        # load the csv files
        behaviour_data = {}
        for i, f in enumerate(csv_files):
            behaviour_data_temp = load_behaviour_file(f)
            trial_time = behaviour_data_temp.name
            behaviour_data[trial_time] = behaviour_data_temp

        # save the behaviour data to a pickle file
        save_pickle(behaviour_data, 'behaviour_data', behaviour_dir)
        
        # split the behaviour data by goal and save to a pickle file
        behaviour_data_by_goal = \
            split_behaviour_data_by_goal(behaviour_data)
        save_pickle(behaviour_data_by_goal, 'behaviour_data_by_goal', behaviour_dir)


    #################### CODE BLOCK 5 ##################
    #################### CALCULATE ADDITIONAL POSITION AND DIRECTION DATA ################
    if starting_code_block <= 5:
        # get the goal coordinates
        screen_coordinates, save_flag = get_screen_coordinates(data_dir)
        # save the screen coordinates
        if save_flag:
            save_pickle(screen_coordinates, 'screen_coordinates', dlc_dir)    
    
        # load dlc_data which has the trial times    
        dlc_data = load_pickle('dlc_final', dlc_dir)
        
        # load the platform coordinates, from which we can get the goal coordinates
        robot_maze_dir = get_robot_maze_directory()
        platform_path = os.path.join(robot_maze_dir, 'workstation', 'map_files')
        platform_coordinates = load_pickle('platform_coordinates', platform_path)
        crop_coordinates = load_pickle('crop_coordinates', platform_path)

        platform_coordinates, save_flag = get_uncropped_platform_coordinates(platform_coordinates, crop_coordinates)
        if save_flag:
            # first, copy the original platform_coordinates file
            src_file = os.path.join(robot_maze_dir, 'workstation',
                    'map_files', 'platform_coordinates.pickle')
            dst_file = os.path.join(robot_maze_dir, 'workstation',
                    'map_files', 'platform_coordinates_cropped.pickle')
            shutil.copyfile(src_file, dst_file)     

            # save the new platform_coordinates file
            save_pickle(platform_coordinates, 'platform_coordinates', platform_path)

        # load the behaviour data, from which we can get the goal ids
        behaviour_dir = os.path.join(data_dir, 'behaviour')
        behaviour_data = load_pickle('behaviour_data_by_goal', behaviour_dir)

        goals = []
        for k in behaviour_data.keys():
            goals.append(k)

        # calculate the animal's current platform for each frame
        dlc_data = get_current_platform(dlc_data, platform_coordinates)  

        # calculate head direction relative to the goals
        dlc_data, goal_coordinates = get_relative_head_direction(dlc_data, platform_coordinates, goals, screen_coordinates)

        # calculate the distance to each goal and screen
        dlc_data = get_distances(dlc_data, platform_coordinates, goal_coordinates, screen_coordinates)

        # save the dlc_data
        save_pickle(dlc_data, 'dlc_final', dlc_dir)


    ####################### CREATE VIDEOS FROM DLC DATA ############################
    # do this in a separate script


    ##################### CODE BLOCK 6 ############################
    ####################### LOAD SORTED SPIKES AND RESTRICT TO TRIALS ###########################
    if starting_code_block <= 6:
        spike_dir = os.path.join(data_dir, 'spike_sorting')
        units = load_sorted_spikes(spike_dir)
        save_pickle(units, 'unit_spike_times', spike_dir)

        dlc_data = load_pickle('dlc_final', dlc_dir)

        # load the spike data
        unit_dir = os.path.join(data_dir, 'spike_sorting')

        units = load_pickle('unit_spike_times', unit_dir)

        restricted_units = restrict_spikes_to_trials(units, dlc_data)
        save_pickle(restricted_units, 'restricted_units', unit_dir)


    ##################### CLASSIFY UNITS ###################################
    # do this in a separate script


    ##################### CODE BLOCK 7 ############################
    ############################### CALCULATE OCCUPANCY MAPS ############################
    if starting_code_block <= 7:
        # load the platform coordinates, from which we can get the goal coordinates
        robot_maze_dir = get_robot_maze_directory()
        platform_dir = os.path.join(robot_maze_dir, 'workstation', 'map_files')
        platform_coordinates = load_pickle('platform_coordinates', platform_dir)

        # get goal coordinates 
        goal_coordinates = get_goal_coordinates(data_dir=data_dir)

        # calculate frame intervals
        dlc_data = calculate_frame_durations(dlc_data)

        # concatenate dlc_data
        dlc_data_concat = concatenate_dlc_data(dlc_data)

        # get axes limits
        limits = get_axes_limits(dlc_data_concat)

        # calculate positional occupancy
        positional_occupancy = get_positional_occupancy(dlc_data_concat, limits)
        # save the positional_occupancy
        save_pickle(positional_occupancy, 'positional_occupancy', dlc_dir)
        # positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

        # plot the the trial paths
        for d in dlc_data.keys():
            # plot_trial_path(dlc_data[d], limits, dlc_dir, d)
            pass

        # plot the heat map of occupancy
        plot_occupancy_heatmap(positional_occupancy, goal_coordinates, dlc_dir)
        
        # calculate directional occupancy
        directional_occupancy = get_directional_occupancy_from_dlc(dlc_data_concat)  
        # save the directional_occupancy
        save_pickle(directional_occupancy, 'directional_occupancy', dlc_dir)

        # plot the directional occupancy
        figure_dir = os.path.join(dlc_dir, 'directional_occupancy_plots')
        plot_directional_occupancy(directional_occupancy, figure_dir)

        # get directional occupancy by position
        directional_occupancy_by_position = get_directional_occupancy_by_position(dlc_data_concat, limits)
        # save the directional_occupancy_by_position
        save_pickle(directional_occupancy_by_position, 'directional_occupancy_by_position', dlc_dir)

        # calculate occupancy by goal
        behaviour_dir = get_behaviour_dir(data_dir)
        behaviour_data = load_pickle('behaviour_data_by_goal', behaviour_dir)
        dlc_data_concat_by_goal = concatenate_dlc_data_by_goal(dlc_data, behaviour_data)
        # save dlc_data_concat_by_goal
        save_pickle(dlc_data_concat_by_goal, 'dlc_data_concat_by_goal', dlc_dir)

        positional_occupancy_by_goal = {}
        directional_occupancy_by_goal = {}

        for g in behaviour_data.keys():
            
            # calculate positional occupancy
            positional_occupancy_by_goal[g] = \
                get_positional_occupancy(dlc_data_concat_by_goal[g], limits)
            
            figure_dir = os.path.join(dlc_dir, 'positional_occupancy_by_goal', f'goal_{g}')
            plot_occupancy_heatmap(positional_occupancy_by_goal[g], goal_coordinates, figure_dir)

            # calculate directional occupancy
            directional_occupancy_by_goal[g] = \
                get_directional_occupancy_from_dlc(dlc_data_concat_by_goal[g])  
            
            figure_dir = os.path.join(dlc_dir, 'directional_occupancy_by_goal', f'goal_{g}')        
            plot_directional_occupancy(directional_occupancy_by_goal[g], figure_dir)
    
        # save the positional_occupancy_by_goal
        save_pickle(positional_occupancy_by_goal, 'positional_occupancy_by_goal', dlc_dir)
        # save the directional_occupancy_by_goal
        save_pickle(directional_occupancy_by_goal, 'directional_occupancy_by_goal', dlc_dir)


    ##################### CODE BLOCK 8 ############################
    ############################## CALCULATE SPIKE POSITIONS AND DIRECTIONS ######################
    if starting_code_block <= 8:
        # load spike data
        spike_dir = os.path.join(data_dir, 'spike_sorting')
        restricted_units = load_pickle('restricted_units', spike_dir)

        # load neuron classification data
        # neuron_types = load_pickle('neuron_types', spike_dir)

        # load positional data
        dlc_dir = os.path.join(data_dir, 'deeplabcut')
        dlc_data = load_pickle('dlc_final', dlc_dir)

        # loop through units and calculate positions and various directional correlates
        for unit in restricted_units.keys():
            restricted_units[unit] = get_unit_position_and_directions(dlc_data, restricted_units[unit])

        # # save the restricted units
        save_pickle(restricted_units, 'units_w_behav_correlates', spike_dir)

        # bin spikes by position
        positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

        # find any NaNs or infs in the positional occupancy['occupancy']
        if np.isnan(positional_occupancy['occupancy']).sum() > 0:
            raise ValueError('There are NaNs in the positional occupancy.')
        if np.isinf(positional_occupancy['occupancy']).sum() > 0:
            raise ValueError('There are infs in the positional occupancy.')


        # load units
        units = load_pickle('units_w_behav_correlates', spike_dir)
        # bin spikes by position
        rate_maps = bin_spikes_by_position(units, positional_occupancy)
        # save the spike counts by position
        save_pickle(rate_maps, 'rate_maps', spike_dir)

        # create smoothed rate_maps
        smoothed_rate_maps = smooth_rate_maps(rate_maps)
        save_pickle(smoothed_rate_maps, 'smoothed_rate_maps', spike_dir) 

        # bin spikes by direction
        directional_occupancy = load_pickle('directional_occupancy', dlc_dir)
        spike_rates_by_direction, spike_counts = bin_spikes_by_direction(units, 
                                                directional_occupancy)
        # save the spike counts and rates by direction
        save_pickle(spike_rates_by_direction, 'spike_rates_by_direction', spike_dir)
        save_pickle(spike_counts, 'spike_counts_by_direction', spike_dir)

        # load the directional occupancy by position data
        directional_occupancy_by_position = load_pickle('directional_occupancy_by_position', dlc_dir)
        # bin spikes by position and direction
        spike_rates_by_position_and_direction, bad_vals = bin_spikes_by_position_and_direction(units, 
                                                directional_occupancy_by_position)
        
        check_bad_vals(bad_vals, dlc_data)


        # save the spike rates by position and direction
        save_pickle(spike_rates_by_position_and_direction, 'spike_rates_by_position_and_direction', spike_dir)

        # create an artificial unit for testing vector field code
        # artifial_unit = create_artificial_unit(units, directional_occupancy_by_position)

        # sort spike data by goal
        behaviour_dir = get_behaviour_dir(data_dir)
        behaviour_data_by_goal = load_pickle('behaviour_data_by_goal', behaviour_dir)

        units_by_goal = sort_units_by_goal(behaviour_data_by_goal, units)

        # load the positional occupancy data by goal
        positional_occupancy_by_goal = load_pickle('positional_occupancy_by_goal', dlc_dir)

        # bin spikes by position by goal
        rate_maps_by_goal = {}
        smoothed_rate_maps_by_goal = {}
        for g in units_by_goal.keys():
            rate_maps_by_goal[g] = bin_spikes_by_position(units_by_goal[g], positional_occupancy_by_goal[g])
            smoothed_rate_maps_by_goal[g] = smooth_rate_maps(rate_maps_by_goal[g])

        save_pickle(rate_maps_by_goal, 'rate_maps_by_goal', spike_dir)  
        save_pickle(smoothed_rate_maps_by_goal, 'smoothed_rate_maps_by_goal', spike_dir)

        # load the directional occupancy data by goal
        directional_occupancy_by_goal = load_pickle('directional_occupancy_by_goal', dlc_dir)

        # bin spikes by direction by goal
        spike_rates_by_direction_by_goal = {}
        spike_counts_by_direction_by_goal = {}
        for g in units_by_goal.keys():
            spike_rates_by_direction_by_goal[g], spike_counts_by_direction_by_goal[g]\
                = bin_spikes_by_direction(units_by_goal[g], directional_occupancy_by_goal[g])

        save_pickle(spike_rates_by_direction_by_goal, 'spike_rates_by_direction_by_goal', spike_dir)


    ########################## CODE BLOCK 9 ############################
    ############################# PLOT PLACE FIELDS ################################
    if starting_code_block <= 9:
        # get goal coordinates
        goal_coordinates = get_goal_coordinates(data_dir=data_dir)

        # load positional data
        dlc_dir = os.path.join(data_dir, 'deeplabcut')
        dlc_data = load_pickle('dlc_final', dlc_dir)

        # get x and y limits
        x_and_y_limits = get_x_and_y_limits(dlc_data)

        # load the positional occupancy data
        positional_occupancy = load_pickle('positional_occupancy', dlc_dir)

        # load the directional occupancy data
        directional_occupancy = load_pickle('directional_occupancy', dlc_dir)

        # load the spike data
        spike_dir = os.path.join(data_dir, 'spike_sorting')
        units = load_pickle('units_w_behav_correlates', spike_dir)

        # plot spikes and position
        plot_dir = os.path.join(spike_dir, 'spikes_and_pos')
        plot_spikes_and_pos(units, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)

        # plot spike and position by goal
        units_by_goal = {}
        for u in units.keys():
            units_by_goal[u] = split_dictionary_by_goal(units[u], data_dir)
        
        plot_dir = os.path.join(spike_dir, 'spikes_and_pos_by_goal')
        plot_spikes_2goals(units_by_goal, dlc_data, goal_coordinates, x_and_y_limits, plot_dir)

        # plot spike rates by direction
        plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction')
        spike_rates_by_direction = load_pickle('spike_rates_by_direction', spike_dir)

        plot_spike_rates_by_direction(spike_rates_by_direction, plot_dir)

        # plot rate maps 
        plot_dir = os.path.join(spike_dir, 'rate_maps')
        rate_maps = load_pickle('rate_maps', spike_dir)
        smoothed_rate_maps = load_pickle('smoothed_rate_maps', spike_dir)  
        plot_rate_maps(rate_maps, smoothed_rate_maps, goal_coordinates, plot_dir)

        # plot smoothed rate maps by goal
        plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal')
        rate_maps_by_goal = load_pickle('smoothed_rate_maps_by_goal', spike_dir)
        plot_rate_maps_2goals(rate_maps_by_goal, goal_coordinates, plot_dir)

        # plot spike rates by direction by goal
        spike_rates_by_direction = load_pickle('spike_rates_by_direction_by_goal', spike_dir)
        plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction_by_goal')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        plot_spike_rates_by_direction_2goals(spike_rates_by_direction, plot_dir)

        for g in spike_rates_by_direction.keys():
            plot_dir = os.path.join(spike_dir, 'spike_rates_by_direction_by_goal', f'goal_{g}')
            plot_spike_rates_by_direction(spike_rates_by_direction[g], plot_dir)

        # plot rate maps by goal
        rate_maps_by_goal = load_pickle('rate_maps_by_goal', spike_dir)
        smoothed_rate_maps_by_goal = load_pickle('smoothed_rate_maps_by_goal', spike_dir)

        plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal')
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        for g  in rate_maps_by_goal.keys():
            plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal', f'goal_{g}')     
            plot_rate_maps(rate_maps_by_goal[g], smoothed_rate_maps_by_goal[g], goal_coordinates, plot_dir)

        
        # create combined plots, with goal 1 in subplot 1 and goal 2 in subplot 2
        plot_dir = os.path.join(spike_dir, 'rate_maps_by_goal', 'both_goals') 
        plot_rate_maps_2goals(rate_maps_by_goal, goal_coordinates, plot_dir)    

        

        
    pass


