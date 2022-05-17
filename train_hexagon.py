import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from functions import gaussian, euclidean_distance, gaussian_second_derivative
from network_higher_dims import Network

hexagon_data = pd.read_csv('data\\hexagon.csv')
x = np.array(hexagon_data.x)
y = np.array(hexagon_data.y)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
hex_xy = np.concatenate((x, y), axis=1)
c = np.array(hexagon_data.c)
c = np.reshape(c, (-1, 1))
hex_data = np.concatenate((hex_xy, c), axis=1)

plt.scatter(x, y, c=c)
plt.title('Hexagon data')
plt.savefig('hexagon_plot.png')
plt.clf()

shapes = [(2, 3), (1, 6)]
neighb_functions = [gaussian, gaussian_second_derivative]
neighb_scales = [1, 0.7, 0.5]

for s in shapes:
    for ns in neighb_scales:
        for i in range(5):
            KN = Network(input_shape=2, shape=s, name=f'{s}_{ns}_cv_{i}')
            KN.fit(hex_xy, 20, neighbourhood_scale=ns, y_train=c.squeeze())

for s in shapes[:2]:
    for ns in neighb_scales:
        for i in range(5):
            KN = Network(input_shape=2, shape=s, name=f'{s}_{ns}_gaussian_second_derivative_cv_{i}',
                         neighbourhood_func=gaussian_second_derivative)
            KN.fit(hex_xy, 20, neighbourhood_scale=ns, y_train=c.squeeze())

