import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.collections import LineCollection
import os
import cebra.datasets
import cebra
import torch
from cebra import CEBRA

sys.path.append('C:/Users/Jake/Documents/python_code/robot_maze_analysis_code')
from utilities.get_directories import get_data_dir


''' using data created with pytorch_decoding.dataset_creation.py '''

if __name__ == "__main__":
    animal = 'Rat46'
    session = '19-02-2024'
    data_dir = get_data_dir(animal, session)

    goal = 52
    window_size = 100

    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    labels_file_name = f'labels_goal{goal}_ws{window_size}'
    # load numpy array of labels
    labels = np.load(os.path.join(dlc_dir, labels_file_name + '.npy'))
    # keep only the first 2 columns
    labels = labels[:, :2]

    # load numpy array of neural data
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    inputs_file_name = f'inputs_goal{goal}_ws{window_size}'
    inputs = np.load(os.path.join(spike_dir, inputs_file_name + '.npy'))

    # load convert inputs to torch tensor
    inputs = torch.tensor(inputs, dtype=torch.float32)


    max_iterations = 10000 #default is 5000.

    cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

    #
    cebra_posdir3_model.fit(inputs, labels)
    
    cebra_model_dir = os.path.join(data_dir, 'cebra')
    # cebra_file_name = f'cebra_posdir3_model_{window_size}.pt'
    cebra_file_name = 'cebra_posdir3_model.pt'
    #cebra_file_path = 'D:/analysis/Rat46/19-02-2024/cebra/cebra_posdir3_model.pt'
    cebra_file_path = os.path.join(cebra_model_dir, cebra_file_name)
    # need to convert to double backslashes
    cebra_posdir3_model.save(cebra_model_dir)

    # cebra_posdir3_model_loaded = cebra.CEBRA.load(cebra_file_path)

    cebra_posdir3 = cebra_posdir3_model.transform(inputs)

    

    # fig = plt.figure(figsize = (7,3), dpi = 100)

    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')

    # ax1 = cebra.plot_embedding(ax=ax1, embedding=cebra_posdir3[::10,:], 
    #     embedding_labels=x, cmap="viridis")

    # ax2 = cebra.plot_embedding(ax=ax2, embedding=cebra_posdir3[::10,:], 
    #     embedding_labels=y, cmap="viridis")

    # ax1.set_title('CEBRA-Dim 3')
    # plt.show()



    def colormap_2d():
        # get the veridis colormap
        v_cmap = plt.get_cmap('viridis')
        v_colormap_values = v_cmap(np.linspace(0, 1, 256))

        # get the cool colormap
        c_cmap = plt.get_cmap('cool')
        c_colormap_values = c_cmap(np.linspace(0, 1, 256))

        # get the indices of each colormap for the 2d map
        v_v, c_v = np.meshgrid(np.arange(256), np.arange(256))

        # create a new 2d array with the values of the colormap
        colormap = np.zeros((256, 256, 4))

        for x in range(256):
            for y in range(256):
                v_val = v_colormap_values[v_v[x, y], :]
                c_val = c_colormap_values[c_v[x, y], :]

                # take the average of the two colormaps
                colormap[x, y, :] = (v_val + c_val) / 2

        return colormap
    

    # x is every 10th value of labels[:,0]
    x = labels[::10,0]
    y = labels[::10,1]

    # convert x and y into vectors of integers between 0 and 255
    x_int = np.interp(x, (x.min(), x.max()), (0, 255)).astype(int)
    y_int = np.interp(y, (y.min(), y.max()), (0, 255)).astype(int)

    embedding_data = cebra_posdir3[::10,:]

    colormap = colormap_2d()  
    # plt.imshow(colormap)  
    color_data = colormap[x_int, y_int]

    fig = plt.figure(figsize = (24,4), dpi = 100)
    ax1 = fig.add_subplot(151, projection='3d')
    # plot embedding_data using the color_data
    ax1.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

    ax2 = fig.add_subplot(152, projection='3d')
    # plot embedding_data using the color_data
    ax2.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

    ax3 = fig.add_subplot(153, projection='3d')
    # plot embedding_data using the color_data
    ax3.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

    ax4 = fig.add_subplot(154, projection='3d')
    # plot embedding_data using the color_data
    ax4.scatter(embedding_data[:,0], embedding_data[:,1], embedding_data[:,2], c=color_data, s=1)

    # for the second subplot: plt.imshow(data)
    ax5 = fig.add_subplot(155)
    ax5.imshow(colormap)

    fig_name = f'goal{goal}_ws{window_size}'
    
    # save fig to the cebra directory
    fig_dir = os.path.join(data_dir, 'cebra')
    fig_path = os.path.join(fig_dir, fig_name + '.png')
    plt.savefig(fig_path)

    pass




