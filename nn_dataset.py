import torch
from torch.utils.data import Dataset

class NNDataset(Dataset):
    """Neural network dataset."""
    # param x: (tensor) input data.
    # param y: (tensor) labels.
    

    def __init__(self, spike_trains, positional_trains, window_edges, transform=None):
        """
        Args:
            spike_trains (dict): Dictionary of spike trains.
            positional_trains (dict): Dictionary of positional trains.
            window_edges (dict): Dictionary of window edges.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.spike_trains = spike_trains
        self.positional_trains = positional_trains
        self.window_edges = window_edges
        self.transform = transform

    def __len__(self):
        return len(self.spike_trains)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'spike_trains': self.spike_trains[idx],
                  'positional_trains': self.positional_trains[idx],
                  'window_edges': self.window_edges[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample