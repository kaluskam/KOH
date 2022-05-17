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
from sklearn.manifold import TSNE


class Network:
    def __init__(self, input_shape, shape, name, metric=euclidean_distance,
                 neighbourhood_func=gaussian, min_val=-1, max_val=1, grid_type='rect'):
        self.grid_type = grid_type
        self.metric = metric
        self.neighbourhood_func = neighbourhood_func
        self.input_shape = input_shape
        self.shape = shape
        self.name = name
        self.weights = None
        self.neurons = None
        self.x_train = None
        self.neighbourhood_scale = None
        self.data = None
        self.learning_rate = None
        self.n_epochs = None
        self.clusters = None
        self.y_train = None
        self.history = pd.DataFrame({'epoch': [], 'adjusted rand score': []})

        if not os.path.isdir(f'evaluation\\{name}'):
            os.mkdir(f'evaluation\\{name}')

        self._initialize_weights(min_val, max_val)

    def _initialize_weights(self, min_val, max_val):
        self.weights = np.random.uniform(min_val, max_val, size=[self.shape[0], self.shape[1], self.input_shape])

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
        if t % 5 == 0:
            evaluation = self.evaluate()
            evaluation['epoch'] = t
            print(f'Epoch no. {t}: {evaluation}')
            self.history = pd.concat([self.history, pd.DataFrame(evaluation, index=[0])])
            self.plot(t)

    def fit(self, x_train, n_epochs, neighbourhood_scale, start_epoch=0, learning_rate=1, y_train=None):
        print(f'Fitting network with name: {self.name}.')
        self.n_epochs = n_epochs
        self.neighbourhood_scale = neighbourhood_scale
        self.learning_rate = learning_rate
        self.x_train = x_train
        self.y_train = y_train

        for t in range(start_epoch, n_epochs):
            permuted_data = np.random.permutation(self.x_train)
            self.report(t)
            for num, x in enumerate(permuted_data):
                BMU_coord = self._find_BMU(x)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        n1 = np.array(BMU_coord)
                        n2 = np.array([i, j])
                        nw = self.neighbourhood_weights(n1, n2, t)
                        delta_weights = nw * self.decay_function(t) * (x - self.weights[i, j, :])
                        self.weights[i, j, :] += delta_weights
        self.save_weights()
        self.report(n_epochs)
        self.save_history()

    def cluster(self):
        centers = np.reshape(self.weights, (-1, self.input_shape))
        self.clusters = []
        for observation in self.x_train:
            distances = np.array(
                [np.linalg.norm(observation - center) for center in centers])
            self.clusters.append(np.argmin(distances))

    def evaluate(self):
        self.cluster()
        return {'adjusted rand score': adjusted_rand_score(self.y_train, self.clusters)}

    def plot(self, epoch=''):
        x_train_embedded = TSNE(n_components=2).fit_transform(self.x_train)
        plt.scatter(x_train_embedded[:, 0], x_train_embedded[:, 1], c=self.clusters)
        plt.savefig(f'evaluation\\{self.name}\\epoch_{epoch}.png')
        plt.clf()

    def save_weights(self):
        np.save(f'weights\\{self.name}.npy', self.weights)

    def load_weights(self):
        self.weights = np.load(f'weights\\{self.name}.npy')

    def save_history(self):
        self.history.to_csv(f'evaluation\\{self.name}.csv')
