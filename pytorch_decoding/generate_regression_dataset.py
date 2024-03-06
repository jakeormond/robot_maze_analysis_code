from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import os
import numpy as np

# generate regression dataset
n_samples = 10000
n_features = 100
n_informative = 1
n_targets = 10

features, target = make_regression(n_samples=n_samples,
                                  n_features=n_features,
                                  n_informative=n_informative,
                                  n_targets=n_targets,
                                  random_state=42)

# convert to float32
features = features.astype(np.float32)
target = target.astype(np.float32)

# plot the regression dataset in 3d
# plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(features[:, 0], features[:, 1], target)
# plt.show()


# plt.scatter(features, target)
# plt.show()

# save the dataset
data_dir = '/media/jake/DataStorage_6TB/DATA/neural_network'
np.save(os.path.join(data_dir, 'features.npy'), features)
np.save(os.path.join(data_dir, 'target.npy'), target)





pass

