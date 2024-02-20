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


# generate regression dataset
X, y = make_regression(
    n_samples=1000, n_features=10, noise=5, random_state=4)

y = y.reshape(-1, 1)

X_scaler = MinMaxScaler()
X_scaled = X_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(
#     X_scaled, y_scaled, test_size=0.33, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.33, random_state=42)






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
    # model.train()
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
  
  # generate regression dataset
  X, y = make_regression(
      n_samples=1000, n_features=10, noise=5, random_state=4)

  y = y.reshape(-1, 1)

  X_scaler = MinMaxScaler()
  X_scaled = X_scaler.fit_transform(X)
  y_scaler = MinMaxScaler()
  y_scaled = y_scaler.fit_transform(y)

  # X_train, X_test, y_train, y_test = train_test_split(
  #     X_scaled, y_scaled, test_size=0.33, random_state=42)

  X_train, X_test, y_train, y_test = train_test_split(
      X_scaled, y_scaled, test_size=0.33, random_state=42)

  n_units = X_train.shape[1]
  
  n_outputs = y_train.shape[1]   
  
  # create loss function and optimizer
  learning_rate = 0.01
  batch_size = 64
  n_epochs = 1000


  class Data(Dataset):
      def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
          # need to convert float64 to float32 else
          # will get the following error
          # RuntimeError: expected scalar type Double but found Float
          self.X = torch.from_numpy(X.astype(np.float32))
          self.y = torch.from_numpy(y.astype(np.float32))
          self.len = self.X.shape[0]
      def __getitem__(self, index: int) -> tuple:
          return self.X[index], self.y[index]
      def __len__(self) -> int:
          return self.len

  training_data = Data(X_train, y_train)


  # training_data = NNDataset(x_train, y_train)
  # testing_data = NNDataset(x_test, y_test)
  train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  # test_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

  # initialize the model
  model = SeqNet(n_units, 64, n_outputs).to(device)
  print(model)

  # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

  loss_fn = nn.MSELoss()

  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  history = []
  epochs = n_epochs
  for t in range(epochs):
      
      # train_loop(train_dataloader, model, loss_fn, optimizer, device)
      # test_loop(test_dataloader, model, loss_fn, device)
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(train_dataloader):
          # forward propagation
          outputs = model(inputs.to(device))
          loss = loss_fn(outputs, labels.to(device))
          # set optimizer to zero grad
          # to remove previous epoch gradients
          optimizer.zero_grad()
          # backward propagation
          loss.backward()
          # optimize
          optimizer.step()
          running_loss += loss.item()

      y_pred = model(X_test.to(device))
      mse = loss_fn(y_pred, y_test.to(device))
      history.append(mse.detach().cpu().numpy())

      # display statistics
      if not ((t + 1) % (epochs // 10)):
          print(f'Epochs:{t + 1:5d} | '
                      f'Batches per epoch: {i + 1:3d} | '
                      f'Loss: {running_loss / (i + 1):.10f}')

  print("Done!")

  # plot the loss
  plt.plot(history)
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.show()    

  figure = plt.figure()
  y_pred = model(X_test.to(device))
  plt.scatter(y_pred.detach().cpu().numpy(), y_test.detach().cpu().numpy())
  plt.xlabel('y_pred')
  plt.ylabel('y_test')
  plt.show()

  
pass