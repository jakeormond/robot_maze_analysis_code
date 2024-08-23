import sys
sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')

# run extract_pulses_from_raw.py, located in the utilities directory
import utilities.extract_pulses_from_raw as epr
import utilities.get_pulses as gp
import utilities.get_video_endpoints as gve

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
import spikes.calculate_consinks as csk


if __name__ == "__main__":
    
    experiment = 'robot_single_goal'
    animal = 'Rat_HC3'
    session = '23-07-2024'

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
    
    