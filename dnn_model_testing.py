import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime
import matplotlib.pyplot as plt

from get_directories import get_data_dir
from load_and_save_data import load_pickle, save_pickle
from nn_dataset import NNDataset

############# HYPEPARAMETERS #########################
learning_rate = 0.01
batch_size = 64
n_epochs = 1000
n_hidden = 64
loss_fn = nn.MSELoss()


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
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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

    test_loss /= num_batches
    print(f"Avg loss: {test_loss:>8f} \n")

    return test_loss


if __name__ == "__main__":
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using {device} device")
    if device == "cuda":
        print(torch.cuda.get_device_name(0))
    else:
        # throw error if not using GPU  
        raise Exception("Need a GPU to train the model")    


    ########### generate regression dataset ##########
    X, y = make_regression(
        n_samples=1000, n_features=100, n_informative= 50, n_targets = 10, noise=5, random_state=4)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    
    ######## scale the data ###########################
    X_scaler = MinMaxScaler()
    X_scaled = X_scaler.fit_transform(X)
    y_scaler = MinMaxScaler()
    y_scaled = y_scaler.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.33, random_state=42)
    
    ############# create dataloaders ##################
    training_data = NNDataset(X_train, y_train)
    testing_data = NNDataset(X_test, y_test)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    # create test tensors for final plot
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    ############### create the model ####################
    n_units = X_train.shape[1]    
    n_outputs = y_train.shape[1]   
    model = SeqNet(n_units, n_hidden, n_outputs).to(device)
    print(model)


    ########## OPTIMIZER ################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

 
    ############# train the model with kfolds #####################
        n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)
    k_fold_splits = {}
    history = []





    epochs = n_epochs
    for t in range(epochs):
        
        # print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_dataloader, model, loss_fn, device)
        history.append(test_loss)
            
        
    print("Done!")

    # plot the loss
    plt.plot(history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()    

    figure = plt.figure()
    y_pred = model(X_test.to(device))
    y_pred = y_pred.detach().cpu().numpy()
    y_test = y_test.detach().cpu().numpy()
    plt.scatter(y_pred[:,1], y_test[:, 5])
    plt.xlabel('y_pred')
    plt.ylabel('y_test')
    plt.show()

    
pass