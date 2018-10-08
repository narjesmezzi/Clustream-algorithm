from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from sklearn.streams.model import MicroCluster as model
from scipy.spatial import distance
import math
import numpy as np
import threading
import time
import sys



class CluStream(BaseEstimator, ClusterMixin):
    #Implementation of CluStream

    def __init__(self,  nb_initial_points=1000, time_window=1000, timestamp=0, clocktime=0, nb_micro_cluster=100,
                nb_macro_cluster=5, micro_clusters=[], alpha=2, l=2, h=1000):
        self.start_time = time.time()
        self.nb_initial_points = nb_initial_points
        self.time_window = time_window  # Range of the window
        self.timestamp = timestamp
        self.clocktime = clocktime
        self.micro_clusters = micro_clusters
        self.nb_micro_cluster = nb_micro_cluster
        self.nb_macro_cluster = nb_macro_cluster
        self.alpha = alpha
        self.l = l
        self.h = h
        #self.snapshots = []
        self.nb_created_clusters = 0

    def fit(self, X, Y=None):
        X = check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points >= self.nb_initial_points:
            kmeans = KMeans(n_clusters=self.nb_micro_cluster, random_state=1)
            micro_cluster_labels = kmeans.fit_predict(X, Y)
            X = np.column_stack((micro_cluster_labels, X))
            initial_clusters = [X[X[:, 0] == l][:, 1:] for l in set(micro_cluster_labels) if l != -1]
            for cluster in initial_clusters:
                self.create_micro_cluster(cluster)
        self.start_time = time.time()

    def create_micro_cluster(self, cluster):
        linear_sum = np.zeros(cluster.shape[1])
        squared_sum = np.zeros(cluster.shape[1])
        self.nb_created_clusters += 1
        new_m_cluster = model(identifier=self.nb_created_clusters, nb_points=0, linear_sum=linear_sum,
                                        squared_sum=squared_sum, update_timestamp=0)
        for point in cluster:
            new_m_cluster.insert(point, self.timestamp)
        self.micro_clusters.append(new_m_cluster)

    # take a snapshot at each second
    """
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
    """

    def distance_to_cluster(self, x, cluster):
        return distance.euclidean(x, cluster.get_center())

    def find_closest_cluster(self, x, micro_clusters):
        min_distance = sys.float_info.max
        for cluster in micro_clusters:
            distance_cluster = self.distance_to_cluster(x, cluster)
            if distance_cluster < min_distance:
                min_distance = distance_cluster
                closest_cluster = cluster
        return closest_cluster

    def check_fit_in_cluster(self, x, cluster):
        if cluster.get_weight() == 1:
            # determine radius using next closest micro-cluster
            radius = sys.float_info.max
            micro_clusters = self.micro_clusters.copy()
            micro_clusters.remove(cluster)
            next_cluster = self.find_closest_cluster(x, micro_clusters)
            dist = distance.euclidean(next_cluster.get_center(), cluster.get_center())
            radius = min(dist, radius)
        else:
            radius = cluster.get_radius()
        if self.distance_to_cluster(x, cluster) < radius:
            return True
        else:
            return False

    def oldest_updated_cluster(self):
        threshold = self.timestamp - self.time_window
        min_relevance_stamp = sys.float_info.max
        oldest_cluster = None
        for cluster in self.micro_clusters:
            relevance_stamp = cluster.get_relevancestamp()
            if (relevance_stamp < threshold) and (relevance_stamp < min_relevance_stamp):
                min_relevance_stamp = relevance_stamp
                oldest_cluster = cluster
        return oldest_cluster

    def merge_closest_clusters(self):
        min_distance = sys.float_info.max
        for i, cluster in enumerate(self.micro_clusters):
            center = cluster.get_center()
            for next_cluster in self.micro_clusters[i+1:]:
                dist = distance.euclidean(center, next_cluster.get_center())
                if dist < min_distance:
                    min_distance = dist
                    cluster_1 = cluster
                    cluster_2 = next_cluster
        assert (cluster_1 != cluster_2)
        cluster_1.merge(cluster_2)
        self.micro_clusters.remove(cluster_2)

    def partial_fit(self, x, y):
        self.timestamp += 1
        X = x
        x = x[0]
        closest_cluster = self.find_closest_cluster(x, self.micro_clusters)
        check = self.check_fit_in_cluster(x, closest_cluster)
        if check:
            closest_cluster.insert(x, self.timestamp)
        else:
            old_up_clust = self.oldest_updated_cluster()
            if old_up_clust is not None:
                self.micro_clusters.remove(old_up_clust)
            else:
                self.merge_closest_clusters()
            self.create_micro_cluster(X)


    def predict(self, X=None):
        """Predict the class labels for the provided data
        Parameters
        ----------
        X :
        Returns
        -------
        y :
        """
        cluster_centers = list(map((lambda i: i.get_center()), self.micro_clusters))
        #centers_weights = list(map((lambda i: i.get_weight()), self.micro_clusters))
        kmeans = KMeans(n_clusters=self.nb_macro_cluster, random_state=1)
        result = kmeans.fit_predict(X=cluster_centers, y=None)
        return result
