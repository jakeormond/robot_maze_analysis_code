import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from sklearn.model_selection import KFold
import datetime

from get_directories import get_data_dir
from load_and_save_data import load_pickle, save_pickle
from nn_dataset import NNDataset

class SeqNet(nn.Module):

    def __init__(self, n_units, n_hidden, n_outputs): # n_units is the number of units in the spike train
        super(SeqNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_units, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_outputs),
        )

    def forward(self, inputs):
        outputs = self.linear_relu_stack(inputs)
        return outputs


# train and test loops
def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    test_loss = 0
    num_batches = len(dataloader)

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # torch.set_default_device(device)

    print(f"Using {device} device")
    if device == "cuda":
        print(torch.cuda.get_device_name(0))
    else:
        # throw error if not using GPU  
        raise Exception("Need a GPU to train the model")

    # create dataset
    # load the spike data and positional data
    animal = 'Rat65'
    session = '10-11-2023'
    data_dir = get_data_dir(animal, session)    
    
    spike_dir = os.path.join(data_dir, 'spike_sorting')
    # load spike train inputs.npy
    inputs = np.load(f'{spike_dir}/inputs.npy')
    n_units = inputs.shape[1]

    # load position train labels.npy
    dlc_dir = os.path.join(data_dir, 'deeplabcut')
    labels = np.load(f'{dlc_dir}/labels.npy')
    n_outputs = labels.shape[1]   
   
    n_hidden = 512
    model = SeqNet(n_units, n_hidden, n_outputs).to(device)
    print(model)

    # create loss function and optimizer
    learning_rate = 0.0001
    batch_size = 256
    n_epochs = 5

    loss_fn = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train the model using k-fold cross validation with 5 folds
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    k_fold_splits = {}

    for i, (train_index, test_index) in enumerate(kf.split(inputs)):
        k_fold_splits[i] = (train_index, test_index)

        training_data = NNDataset(inputs[train_index, :], labels[train_index, :])
        testing_data = NNDataset(inputs[test_index, :], labels[test_index, :])
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

        # re-initialize the model weights
        model = SeqNet(n_units, n_hidden, n_outputs).to(device)

        epochs = n_epochs
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, device)
            test_loop(test_dataloader, model, loss_fn, device)
        print("Done!")

        # generate path to save the model
        model_dir = os.path.join(data_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        # generate model name so it includes the date and time it was generated
        current_time = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        model_name = f'SeqNet_split{i}_{current_time}.pth'
        model_path = os.path.join(model_dir, model_name)
        torch.save(model, model_path)

    # save the splits
    save_pickle(k_fold_splits, 'k_fold_splits', model_dir)
    pass