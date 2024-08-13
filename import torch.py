import torch

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
print(f'Train on gpu: {train_on_gpu}')

pass