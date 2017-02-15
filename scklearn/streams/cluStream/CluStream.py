from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from scklearn.streams.cluStream import MicroCluster
from scipy.spatial import distance
import math as math
import numpy as np
import threading
import time



class CluStream(BaseEstimator, ClusterMixin):
    #Implementation of CluStream

    def __init__(self,  nb_initial_points = 1000, time_window = 1000, timestamp = 0, clocktime = 0, nbr_cluster = 100
                 , micro_cluster = [], cluster_radisfactor = 2, alpha = 2, l = 2, h = 1000):
        self.start_time = time.time()
        self.nb_initial_points = nb_initial_points
        self.time_window = time_window  # Range of the window
        self.timestamp = timestamp
        self.clocktime = clocktime
        self.micro_cluster = micro_cluster
        self.nbr_cluster = nbr_cluster
        self.cluster_radisfactor = cluster_radisfactor
        self.alpha = alpha
        self.l = l
        self.h = h
        self.snapshot_ct = []

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
        self.start_time = time.time()

    def create_micro_cluster(self, cluster):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        new_m_cluster = self.MicroCluster(nb_points=0, linear_sum=linear_sum, squared_sum=squared_sum,
                                             update_timestamp=0)
        [new_m_cluster.insert(point, self.current_timestamp) for point in cluster]
        self.micro_cluster.append(new_m_cluster)

    # take a snapshot at each second
    def snapshots_taking(self):
        threading.Timer(1.0, self.snapshots_taking).start()
        clock_time = int(round(time.time() - self.start_time))
        print(str(clock_time))
        snapshot = open(str(clock_time) + ".txt", "w")
        snapshot.write(str(self.micro_cluster))
        snapshot.close()

    def snapshot_manager(self):
        # check the number of snapshot per order
        # check if the time of snapshot is redundant
        # delete the convienient snapshot
        max_order = math.pow(self.alpha, self.l) + 1
        nbr_orders = max_order + 1
        nbr_snapshot_order = np.zeros(nbr_orders)

    def find_closest_cluster(self, x) :
        micro_cluster = None
        distance = 1000
        for i in len(self.micro_cluster) :
            if distance.euclidean(x, self.micro_cluster[i].get_center) < distance :
                distance = distance.euclidean(x, self.micro_cluster[i])
                micro_cluster = self.micro_cluster[i]
        return micro_cluster


    def partial_fit(self, x, y) :
        self.timestamp += 1
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
