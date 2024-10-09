import sys
import os

# if on windows system
if os.name == 'nt':
    sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')

# if on linux system
else:
    sys.path.append('/home/jake/Documents/python_code/robot_maze_analysis_code')

import os

# run extract_pulses_from_raw.py, located in the utilities directory
import utilities.extract_pulses_from_raw as epr
import utilities.get_pulses as gp
import utilities.get_video_endpoints as gve
import utilities.get_directories as gd

import position.process_dlc_data as pdd
import position.calculate_pos_and_dir as cpd
import position.create_videos_with_dlc_data as cvd
import position.calculate_occupancy as co

import behaviour.load_behaviour as lb

import spikes.load_sorted_spikes as lss
import spikes.restrict_spikes_to_trials as rst
import spikes.classify_neurons as cn
import spikes.calculate_spike_pos_hd as csp
import spikes.plot_spikes_and_pos as psp
import spikes.plot_channel_map as pcm
import spikes.calculate_consinks as csk
import spikes.calculate_vector_fields as cvf


def main():
    experiment = 'robot_single_goal'
    animal = 'Rat_HC4'
    session = '01-08-2024'

    epr.main(experiment=experiment, animal=animal, session=session)

    gp.main(experiment=experiment, animal=animal, session=session)

    gve.main(experiment=experiment, animal=animal, session=session)

    pdd.main(experiment=experiment, animal=animal, session=session)

    lb.main(experiment=experiment, animal=animal, session=session)

    cpd.main(experiment=experiment, animal=animal, session=session)

    cvd.main(experiment=experiment, animal=animal, session=session, n_trials=2)

    lss.main(experiment=experiment, animal=animal, session=session)

    rst.main(experiment=experiment, animal=animal, session=session)

    cn.main(experiment=experiment, animal=animal, session=session)

    co.main(experiment=experiment, animal=animal, session=session)
    
    csp.main_1goal(experiment=experiment, animal=animal, session=session)

    psp.main(experiment=experiment, animal=animal, session=session)

    csk.main(experiment=experiment, animal=animal, session=session, code_to_run = [0])


def main2():
    experiment = 'robot_single_goal'
    
    animals = ['Rat_HC1', 'Rat_HC2', 'Rat_HC3', 'Rat_HC4']
    # animals = ['Rat_HC3', 'Rat_HC4']
    # animals = ['Rat_HC2', 'Rat_HC3', 'Rat_HC4']

    for animal in animals:

        # find directories in the animal directory
        home_directory = gd.get_home_dir()
        parent_directory = os.path.join(home_directory, experiment, animal)
        
        directories = os.listdir(parent_directory)

        for session in directories:
            cpd.main(experiment=experiment, animal=animal, session=session)
            # pcm.main(experiment=experiment, animal=animal, session=session)
            # csk.main(experiment=experiment, animal=animal, session=session, code_to_run = [9])
            # cvf.main(experiment=experiment, animal=animal, session=session, code_to_run = [1])
            # lb.main3(experiment=experiment, animal=animal, session=session)
            pass


if __name__ == "__main__":
    
    # main()
    main2()
    
    