import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions import gaussian, euclidean_distance
from network import Network

hexagon_data = pd.read_csv('data\\hexagon.csv')
x = np.array(hexagon_data.x)
y = np.array(hexagon_data.y)
x = np.reshape(x, (-1, 1))
y = np.reshape(y, (-1, 1))
hex_xy = np.concatenate((x, y), axis=1)
c = np.array(hexagon_data.c)

KN = Network(input_shape=2, shape=(20, 20))
BMU = KN._find_BMU(hex_xy[0,:])
KN.fit(hex_xy, 2)
clusters = KN.cluster(hexagon_data)
print(clusters)