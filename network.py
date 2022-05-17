import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import re

import pandas as pd

import functions
from functions import gaussian, euclidean_distance, gaussian_second_derivative
from visualisation_utils import *
from sklearn.metrics import adjusted_rand_score


class Network:
    def __init__(self, input_shape, shape, name, metric=euclidean_distance,
                 neighbourhood_func=gaussian, min_val=-1, max_val=1, grid_type='rect'):
        self.grid_type = grid_type
        self.all_data = None
        self.neighbourhood_scale = None
        self.data = None
        self.learning_rate = None
        self.n_epochs = None
        self.metric = metric
        self.neighbourhood_func = neighbourhood_func
        self.input_shape = input_shape
        self.shape = shape
        self.weights = None
        self.neurons = None
        self.name = name
        self._initialize_weights(min_val, max_val)

        self.history = pd.DataFrame({'epoch': [], 'adjusted rand score': []})

        if not os.path.isdir(f'evaluation\\{name}'):
            os.mkdir(f'evaluation\\{name}')

    def _initialize_weights(self, min_val, max_val):
        self.weights = np.random.uniform(min_val, max_val, size=[self.shape[0],
                                                                 self.shape[1],
                                                                 self.input_shape])

    def decay_function(self, t):
        return np.exp(-t / self.n_epochs)

    def distance_between_neighbors(self, n1, n2):
        if self.grid_type == 'hex':
            return functions.hexagonal_distance(n1, n2)

        return functions.rectangular_distance(n1, n2)

    def neighbourhood_weights(self, n1, n2, t):
        distance = self.distance_between_neighbors(n1, n2)
        return self.neighbourhood_func(distance * self.neighbourhood_scale, t)

    def _find_BMU(self, x):
        distances = self.metric(self.weights, x)
        return np.unravel_index(np.argmin(distances, axis=None),
                                distances.shape)

    def report(self, t):
        if t % 5 == 1:
            evaluation = self.evaluate()
            evaluation['epoch'] = t
            print(f'Epoch no. {t}: {evaluation}')
            self.history = pd.concat([self.history, pd.DataFrame(evaluation, index=[0])])
            self.plot(t)

    def fit(self, all_data, n_epochs, neighbourhood_scale, learning_rate=1):
        self.n_epochs = n_epochs
        self.neighbourhood_scale = neighbourhood_scale
        self.learning_rate = learning_rate
        self.all_data = all_data
        self.data = all_data[:, np.arange(self.input_shape)]

        for t in range(n_epochs):
            np.random.shuffle(self.data)
            self.report(t)
            for num, x in enumerate(self.data):
                BMU_coord = self._find_BMU(x)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        n1 = np.array(BMU_coord)
                        n2 = np.array([i, j])
                        nw = self.neighbourhood_weights(n1, n2, t)
                        delta_weights = nw * self.decay_function(t) * (x - self.weights[i, j, :])
                        self.weights[i, j, :] += delta_weights

    def cluster(self, data):
        centers = np.reshape(self.weights, (-1, self.input_shape))
        data_clusters = []
        for row in data:
            distances = np.array(
                [np.linalg.norm(row - center) for center in centers])
            data_clusters.append(np.argmin(distances))
        return data_clusters

    def plot(self):
        if self.is_n_neurons_equal_n_clusters():
            if self.input_shape == 2:
                plot_2D_clustered(self.weights, self.cluster(self.all_data[:, [0, 1]]),
                                  self.all_data)

            if self.input_shape == 3:
                plot_3D_clustered(self.weights, self.cluster(self.all_data[:, [0, 1, 2]]),
                                  self.all_data)

        else:
            if self.input_shape == 2:
                plot_2D_not_clustered(self.weights)

    def is_n_neurons_equal_n_clusters(self):
        n_neurons = 1
        for i in range(len(self.shape)):
            n_neurons *= self.shape[i]
        n_clusters = len(np.unique(self.all_data[:, -1]))
        return n_neurons == n_clusters

    def evaluate(self):
        self.cluster()
        return {'adjusted rand score': adjusted_rand_score(self.y_train,
                                                           self.clusters)}