import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.cluster import KMeans

from visualisation_utils import metric_scores_by_n_clusters
from network import Network


def cross_validate_network(params, fit_params, k=5):
    metrics = [silhouette_score, calinski_harabasz_score]
    scores_mean = {'silhouette_score': None, 'calinski_harabasz_score': None}
    for i in range(k):
        n = Network(**params)
        n.fit(**fit_params)
        for metric in metrics:
            n_clusters_list, score = metric_scores_by_n_clusters(metric,
                                                                 n.weights.reshape((-1, 2)))
            if i == 0:
                scores_mean[metric.__name__] = np.array(score) / k
            else:
                scores_mean[metric.__name__] += np.array(score) / k

    return scores_mean