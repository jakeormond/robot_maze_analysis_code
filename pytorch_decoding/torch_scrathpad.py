import torch
from torch.utils.data import Dataset
import numpy as np

# make an 100 by 10 array of random numbers
data = np.random.rand(100, 10)

# convert the array to a tensor
data = torch.tensor(data, dtype=torch.float32)

pass
