import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import re
from functions import gaussian, euclidean_distance, gaussian_second_derivative


class Network:
    def __init__(self, input_shape, shape, metric=euclidean_distance,
                 neighbourhood_func=gaussian, min_val=-1, max_val=1):
        self.learning_rate = None
        self.n_epochs = None
        self.metric = metric
        self.neighbourhood_func = neighbourhood_func
        self.input_shape = input_shape
        self.shape = shape
        self.weights = None
        self.neurons = None
        self.name = None
        self._initialize_weights(min_val, max_val)

    def _create_name(self):
        if self.name is not None:
            matches = re.search('e=\d+', self.name)
            epochs = int(matches.group(0).replace('e=', ''))
            epochs += self.n_epochs
        else:
            epochs = self.n_epochs
        self.name = f'KN_{self.shape[0]}x{self.shape[1]}_lr={self.learning_rate}_e={epochs}_{self.neighbourhood_func.__name__}'

    def _initialize_weights(self, min_val, max_val):
        self.weights = np.random.uniform(min_val, max_val, size=[self.shape[0],
                                                                 self.shape[1],
                                                                 self.input_shape])

    def decay_function(self, t):
        return np.exp(-t / self.n_epochs)

    def neighbourhood_weights(self, n1, n2, t):
        distance = self.metric(n1, n2)
        return self.neighbourhood_func(distance, t)

    def find_closest_neighbour(self, x):
        min_distance = sys.maxsize
        i_min = 0
        j_min = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                distance = self.metric(self.weights[:, i, j], x)
                if distance < min_distance:
                    min_distance = distance
                    i_min = i
                    j_min = j

        return i_min, j_min

    def _find_BMU(self, x):
        distances = self.metric(self.weights, x)
        return np.unravel_index(np.argmin(distances, axis=None),
                                distances.shape)

    def _update_weights(self, BMU_coord):
        # weigths_delta =
        pass

    def fit(self, data, n_epochs, neighbourhood_scale, learning_rate=1):
        self.n_epochs = n_epochs
        neighbourhood_scale = neighbourhood_scale
        self.learning_rate = learning_rate
        self._create_name()
        for t in range(n_epochs):
            np.random.shuffle(data)
            print(f'Epoch no. {t}')
            for x in data:
                BMU_coord = self._find_BMU(x)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        n1 = np.array(BMU_coord)
                        n2 = np.array([i, j])
                        delta_weights = self.neighbourhood_weights(n1, n2, t) \
                                        * self.learning_rate \
                                        * self.decay_function(t) * (
                                                x * neighbourhood_scale - self.weights[
                                                                          i, j,
                                                                          :])
                        self.weights[i, j, :] += delta_weights
        self.save_weights()

    def cluster(self, data):
        clusters = []
        for i in range(len(data)):
            x = np.array(data.iloc[i, :2])
            i_min, j_min = self._find_BMU(x)
            clusters.append([data.loc[i, 'c'], i_min, j_min])
        return np.array(clusters)

    def visualise(self, clusters):
        x = clusters[:, 1][0]
        y = clusters[:, 1][1]
        colors = clusters[:, 0]
        neurons = np.zeros(shape=self.shape)
        for i in range(len(x)):
            neurons[x[i], y[i]] = colors[i]

        plt.imshow(neurons)
        plt.show()

    def save_weights(self):
        if not os.path.isdir('weights'):
            os.mkdir('weights')
        np.save(os.path.join('weights', self.name), self.weights)

    def load_weights(self, path):
        self.weights = np.load(path)
        self.name = path.replace('weights/', '').replace('.npy', '')
