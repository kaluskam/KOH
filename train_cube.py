import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from functions import gaussian, euclidean_distance, gaussian_second_derivative
from network_higher_dims import Network

cube_data = pd.read_csv('data\\cube.csv')
x = np.array(cube_data.x)
y = np.array(cube_data.y)
z = np.array(cube_data.z)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
z = np.reshape(z, (-1, 1))
cube_xyz = np.concatenate((x, y, z), axis=1)
c = np.array(cube_data.c)
c = np.reshape(c, (-1, 1))
cube_data = np.concatenate((cube_xyz, c), axis=1)

# plt.scatter(x, y, c=c)
# plt.title('cube data')
# plt.savefig('cube_plot.png')
# plt.clf()

shapes = [(2, 4), (1, 8)]
neighb_functions = [gaussian, gaussian_second_derivative]
neighb_scales = [1, 0.7, 0.5]

for s in shapes:
    for ns in neighb_scales:
        for i in range(5):
            KN = Network(input_shape=3, shape=s, name=f'cube_{s}_{ns}_cv_{i}')
            KN.fit(cube_xyz, 20, neighbourhood_scale=ns, y_train=c.squeeze())

for s in shapes[:2]:
    for ns in neighb_scales:
        for i in range(5):
            KN = Network(input_shape=3, shape=s, name=f'cube_{s}_{ns}_gaussian_second_derivative_cv_{i}',
                         neighbourhood_func=gaussian_second_derivative)
            KN.fit(cube_xyz, 20, neighbourhood_scale=ns, y_train=c.squeeze())
