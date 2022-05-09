import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import gaussian, euclidean_distance, gaussian_second_derivative
from network import Network

hexagon_data = pd.read_csv('data\\hexagon.csv')
x = np.array(hexagon_data.x)
y = np.array(hexagon_data.y)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
hex_xy = np.concatenate((x, y), axis=1)
c = np.array(hexagon_data.c)
c = np.reshape(c, (-1, 1))
hex_data = np.concatenate((hex_xy, c), axis=1)

KN = Network(input_shape=2, shape=(2, 3), neighbourhood_func=gaussian_second_derivative)

KN.fit(hex_data, n_epochs=5, neighbourhood_scale=1)
