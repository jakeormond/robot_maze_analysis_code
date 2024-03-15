import matplotlib.pyplot as plt
import numpy as np


# get the veridis colormap
v_cmap = plt.get_cmap('viridis')
v_colormap_values = v_cmap(np.linspace(0, 1, 256))

# get the cool colormap
c_cmap = plt.get_cmap('cool')
c_colormap_values = c_cmap(np.linspace(0, 1, 256))

# get the indices of each colormap for the 2d map
v_v, c_v = np.meshgrid(np.arange(256), np.arange(256))

# create a new 2d array with the values of the colormap
data = np.zeros((256, 256, 4))

for x in range(256):
    for y in range(256):
        v_val = v_colormap_values[v_v[x, y], :]
        c_val = c_colormap_values[c_v[x, y], :]

        # take the average of the two colormaps
        data[x, y, :] = (v_val + c_val) / 2


# plot the 2d colormap
plt.imshow(data)
