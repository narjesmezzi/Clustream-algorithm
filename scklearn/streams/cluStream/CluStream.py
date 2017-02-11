from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from scklearn.streams.cluStream import MicroCluster
import math as math
import numpy as np


class CluStream(BaseEstimator, ClusterMixin):
    #Implementation of CluStream

    def __init__(self,  nb_initial_points = 1000, time_window = 1000, timestamp = 0, nbr_cluster = 100
                 , micro_cluster = [], cluster_radisfactor = 2):
        self.nb_initial_points = nb_initial_points
        self.time_window = time_window  # Range of the window
        self.timestamp = timestamp
        self.micro_cluster = micro_cluster
        self.nbr_cluster = nbr_cluster
        self.cluster_radisfactor = cluster_radisfactor

    def fit(self, X, Y=None):
        # use kmeans to generate the nbr_cluster micro-clusters
        X = check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points > self.init_points_option:
            kmeans = KMeans(n_clusters=self.max_num_kernels, random_state=1)
            m_cluster_labels = kmeans.fit_predict(X, Y)
            X = np.column_stack((m_cluster_labels, X))
            initial_clusters = [X[X[:, 0] == str(l)][:, 1:] for l in set(m_cluster_labels) if l != -1]
            [self.create_micro_cluster(cluster) for cluster in initial_clusters]

    def create_micro_cluster(self, cluster):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        new_m_cluster = self.MicroCluster(nb_points=0, linear_sum=linear_sum, squared_sum=squared_sum,
                                             update_timestamp=0)
        [new_m_cluster.insert(point, self.current_timestamp) for point in cluster]
        self.micro_cluster.append(new_m_cluster)


    def find_closest_cluster(self):
        return self



    def partial_fit(self, x, y):
        return self


    def predict(self, X):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :
        Returns
        -------
        y :
        """
        return self
