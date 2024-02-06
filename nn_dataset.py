import torch
from torch.utils.data import Dataset
import numpy as np
import os
from sklearn.model_selection import KFold

from get_directories import get_data_dir
from load_and_save_data import load_pickle, save_pickle

class NNDataset(Dataset):
    """Neural network dataset."""
    # param x: (tensor) input data.
    # param y: (tensor) labels.
    

    def __init__(self, spike_trains, positional_trains, transform=None):
        """
        Args:
            spike_trains (dict): Dictionary of spike trains.
            positional_trains (dict): Dictionary of positional trains.
            window_edges (dict): Dictionary of window edges.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.spike_trains = torch.from_numpy(spike_trains)
        self.positional_trains = torch.from_numpy(positional_trains)
        self.transform = transform

    def __len__(self):
        return len(self.spike_trains)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = {'spike_trains': self.spike_trains[idx],
        #          'positional_trains': self.positional_trains[idx]}

        spike_train = self.spike_trains[idx]
        positional_train = self.positional_trains[idx]

        if self.transform:
            spike_train = self.transform(spike_train)

        return spike_train, positional_train
    

# load the spike data and positional data
animal = 'Rat65'
session = '10-11-2023'
data_dir = get_data_dir(animal, session)

spike_dir = os.path.join(data_dir, 'spike_sorting')
# load spike train inputs.npy
inputs = np.load(f'{spike_dir}/inputs.npy')

# load position train labels.npy
dlc_dir = os.path.join(data_dir, 'deeplabcut')
labels = np.load(f'{dlc_dir}/labels.npy')

# create the k folds splits
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True)
kf.split(inputs)
for i, (train_index, test_index) in enumerate(kf.split(inputs)):
    training_data = NNDataset(inputs[train_index, :], labels[train_index, :])
    testing_data = NNDataset(inputs[test_index, :], labels[test_index, :])

    # then train the model
