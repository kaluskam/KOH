import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans


def plot_true(data):

    if len(data.shape) == 3:
        plt.scatter(x=data[:, 0], y=data[:, 1], c=data[:, 2])
    elif len(data.shape) == 4:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3])
    plt.title('Prawdziwy podział na klastry')


def plot_predicted_and_true_clusters(network, true_data):
    weights = network.weights

    if len(true_data.shape) == 3:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_true(true_data)
        plt.subplot(1, 2, 2)
        clusters = network.cluster(true_data[:, [0, 1]])
        plot_2D_clustered(weights, clusters, true_data)

    elif len(true_data.shape) == 4:
        plot_true(true_data)
        clusters = network.cluster(true_data[:, [0, 1, 2]])
        plot_3D_clustered(weights, clusters, true_data)
        plt.show()


def plot_2D_not_clustered(weights):
    plot_weights(weights)
    plt.title('Pozycje neuronów')
    plt.show()


def plot_2D_clustered(weights, clusters, data):
    print(f'Adjusted rand score: {adjusted_rand_score(data[:, 2].astype("int"), clusters)}')
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plot_weights(weights)
    plt.title('Podział na klastry według sieci Kohonena')
    plt.show()


def plot_weights(weights):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if len(weights.shape) == 3:
                w = weights[i, j]
                plt.scatter(x=[w[0]], y=[w[1]], color='black', s=50)


def plot_weights_3D(weights, ax):
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if len(weights.shape) == 3:
                w = weights[i, j]
                ax.scatter([w[0]], [w[1]], [w[2]], color='black', s=50)


def plot_3D_clustered(weights, clusters, data):
    print(f'Adjusted rand score: {adjusted_rand_score(data[:, 3].astype("int"), clusters)}')
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=clusters)
    plot_weights_3D(weights, ax)
    plt.title('Podział na klastry według sieci Kohonena')
    plt.show()


def plot_metric_scores_by_n_clusters(metric, weights):
    n_clusters_list, scores = metric_scores_by_n_clusters(metric, weights)
    plt.plot(n_clusters_list, scores)
    plt.title(metric.__name__.replace('_', ' '))


def metric_scores_by_n_clusters(metric, weights):
    n_clusters_list = np.arange(2, 10)
    scores = []
    for n_clusters in n_clusters_list:
        clusters = KMeans(n_clusters=n_clusters).fit(weights)
        scores.append(metric(weights, clusters.labels_))
    return n_clusters_list, scores


def plot_list_of_metrics(metrics, weights):
    n = len(metrics)
    plt.figure(figsize=[n * 6, 4])
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plot_metric_scores_by_n_clusters(metrics[i], weights)
