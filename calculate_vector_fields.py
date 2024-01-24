import os
import numpy as np
import pycircstat as circ
import matplotlib.pyplot as plt


from get_directories import get_data_dir, get_robot_maze_directory
from load_and_save_data import load_pickle, save_pickle
from load_behaviour import get_behaviour_dir

def calculate_vector_fields(spike_rates_by_position_and_direction):

    direction_bins = spike_rates_by_position_and_direction['direction_bins']
    bin_centres = direction_bins[:-1] + np.diff(direction_bins)/2

    vector_fields = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']} 
    mean_resultant_lengths = {'units': {}, 'x_bins': spike_rates_by_position_and_direction['x_bins'], 
                     'y_bins': spike_rates_by_position_and_direction['y_bins'], 
                     'direction_bins': spike_rates_by_position_and_direction['direction_bins']} 
    
    for u in spike_rates_by_position_and_direction['units'].keys():
        rates_by_pos_dir = spike_rates_by_position_and_direction['units'][u]
        array_shape = rates_by_pos_dir.shape

        # initialize vector field as array of nans
        vector_field = np.full(array_shape[0:2], np.nan)
        mrl_field = np.full(array_shape[0:2], np.nan)

        for i in range(array_shape[0]):
            for j in range(array_shape[1]):
                rates = rates_by_pos_dir[i, j, :]
                
                # if any nan values in rates, OR if all values are zero skip
                if np.isnan(rates).any() or np.all(rates == 0):
                    continue

                mean_dir = np.round(circ.mean(bin_centres, rates), 3)
                mrl = np.round(circ.resultant_vector_length(bin_centres, rates), 3)

                if mean_dir > np.pi:
                    mean_dir = mean_dir - 2*np.pi

                vector_field[i, j] = mean_dir
                mrl_field[i, j] = mrl

        vector_fields['units'][u] = vector_field
        mean_resultant_lengths['units'][u] = mrl_field

    return vector_fields, mean_resultant_lengths
    
    
def plot_vector_fields(vector_fields, plot_dir):
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    x_bins = vector_fields['x_bins']
    x_centres = x_bins[:-1] + np.diff(x_bins)/2
    y_bins = vector_fields['y_bins']
    y_centres = y_bins[:-1] + np.diff(y_bins)/2

    for u in vector_fields['units'].keys():
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.set_title(u)
        ax.set_xlabel('x position (cm)')
        ax.set_ylabel('y position (cm)')

        vector_field = vector_fields['units'][u]

        # make first row of vector field all zeros
        vector_field[0, :] = 0
        vector_field[1, :] = np.pi/2
        vector_field[2, :] = np.pi
        vector_field[3, :] = -np.pi
        vector_field[4, :] = -np.pi/2

        # plot vector field
        ax.quiver(x_centres, y_centres, np.cos(vector_field), np.sin(vector_field), color='k', scale=10)
        
        # show fig
        plt.show()
        
        # flip y axis
        ax.invert_yaxis()

        # increase the range of the axes by 10% to make room for the arrows
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        ax.set_xlim(x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range)
        ax.set_ylim(y_lim[0] - 0.1*y_range, y_lim[1] + 0.1*y_range)

        fig.savefig(os.path.join(plot_dir, f'{u}.png'))

        plt.close(fig)






    pass


if __name__ == "__main__":
    animal = 'Rat65'
    session = '10-11-2023'
    data_dir = get_data_dir(animal, session)

    # load spike data
    spike_dir = os.path.join(data_dir, 'spike_sorting')

    # load spike_rates_by_position_and_direction
    spike_rates_by_position_and_direction = load_pickle('spike_rates_by_position_and_direction', spike_dir)

    # vector_fields, mean_resultant_lengths = calculate_vector_fields(spike_rates_by_position_and_direction)
    # save_pickle(vector_fields, 'vector_fields', spike_dir)
    # save_pickle(mean_resultant_lengths, 'mean_resultant_lengths', spike_dir)

    # load vector fields
    vector_fields = load_pickle('vector_fields', spike_dir)

    # plot vector fields
    plot_dir = os.path.join(spike_dir, 'vector_fields')
    plot_vector_fields(vector_fields, plot_dir)
