import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import numpy as np


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

traindata = Data(X_train, y_train)


batch_size = 64
# num_workers = 2
trainloader = DataLoader(traindata, 
                         batch_size=batch_size, 
                         shuffle=True)



class LinearRegression(nn.Module):
  def __init__(self, input_dim: int, 
               hidden_dim: int, output_dim: int) -> None:
    super(LinearRegression, self).__init__()
    self.linear_stack = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),       
        
        # self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        # self.hidden_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        # self.hidden_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        # self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
    )
  def forward(self, inputs):
    x = self.linear_stack(inputs)

    # x = self.input_to_hidden(x)
    # x = self.hidden_layer_1(x)
    # x = self.hidden_layer_2(x)
    # x = self.hidden_to_output(x)
    return x


# number of features (len of X cols)
input_dim = X_train.shape[1]
# number of hidden layers
hidden_layers = 64
# output dimension is 1 because of linear regression
output_dim = 1
# initialize the model

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

model = LinearRegression(input_dim, hidden_layers, output_dim).to(device)
print(model)
'''
# Output:
LinearRegression(   
  (input_to_hidden): Linear(in_features=10, out_features=50, bias=True)   
  (hidden_layer_1): Linear(in_features=50, out_features=50, bias=True)   
  (hidden_layer_2): Linear(in_features=50, out_features=50, bias=True)   
  (hidden_to_output): Linear(in_features=50, out_features=1, bias=True) 
)
'''


# criterion to computes the loss between input and target
criterion = nn.MSELoss()
# optimizer that will be used to update weights and biases
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

history = []

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

epochs = 1000
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        # inputs, labels = data
        # forward propagation
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        # set optimizer to zero grad
        # to remove previous epoch gradients
        optimizer.zero_grad()
        # backward propagation
        loss.backward()
        # optimize
        optimizer.step()
        running_loss += loss.item()

    y_pred = model(X_test.to(device))
    mse = criterion(y_pred, y_test.to(device))
    history.append(mse.detach().cpu().numpy())


    # display statistics
    if not ((epoch + 1) % (epochs // 10)):
       print(f'Epochs:{epoch + 1:5d} | '
                f'Batches per epoch: {i + 1:3d} | '
                f'Loss: {running_loss / (i + 1):.10f}')


# plot the loss
figure = plt.figure()
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