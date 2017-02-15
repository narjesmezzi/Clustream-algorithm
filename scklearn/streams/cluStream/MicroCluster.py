import math as math
import numpy as np


class CluMicroCluster:
    """
    Implementation of the MicroCluster data structure for the CluStream algorithm
    Parameters
    ----------
    :parameter nb_points is the number of points in the cluster
    :parameter id is the identifier of the cluster (take -1 if the cluster result from merging two clusters)
    :parameter merge is used to indicate whether the cluster is resulting from the merge of two existing ones
    :parameter id_list is the id list of merged clusters
    :parameter linear_sum is the linear sum of the points in the cluster.
    :parameter squared_sum is  the squared sum of all the points added to the cluster.
    :parameter linear_time_sum is  the linear sum of all the timestamps of points added to the cluster.
    :parameter squared_time_sum is  the squared sum of all the timestamps of points added to the cluster.
    :parameter m is the number of points considered to determine the relevance stamp of a cluster
    :parameter update_timestamp is used to indicate the last update time of the cluster
    """

    def __init__(self, nb_points = 0, identifier = 0, merge = 0, id_list = None, linear_sum = 0,
                 squared_sum = 0, linear_time_sum = 0, squared_time_sum = 0,
                 m = 100, update_timestamp = 0):
        self.nb_points = nb_points
        self.identifier = identifier
        self.merge = merge
        self.id_list = id_list
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.linear_time_sum = linear_time_sum
        self.squared_time_sum = squared_time_sum
        self.m = m
        self.update_timestamp = update_timestamp
        #self.radius_factor = 1.8


    def get_center(self, cluster):
        center = [ self.linear_sum[i] / self.nb_points for i in range(len(self.linear_sum))]
        return center

    def insert(self, x, current_timestamp):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(x):
            self.linear_sum[i] += x[i]
            self.squared_sum[i] += math.pow(x[i], 2)
            self.linear_time_sum[i] += current_timestamp
            self.squared_time_sum[i] += math.pow(current_timestamp, 2)


    def merge(self, micro_clusters):
        #for cluster in micro_clusters :
        return self

    def get_relevancestamp(self):
        if (self.nb_points < 2 * self.m):
            return self.get_mutime()
        return self.get_mutime() + self.get_sigmatime() * self.get_quantile( self.m /(2 * self.nb_points))

    def get_mutime(self):
        return self.linear_time_sum / self.nb_points

    def get_sigmatime(self):
        return math.sqrt(self.square_time_sum / self.nb_points - math.pow((self.linear_time_sum / self.nb_points), 2))

    def get_quantile(self, x):
        # TODO hold the exception ( x >= 0 && x <= 1 )
        return math.sqrt(2) * self.inverse_error(2 * x - 1)

    def get_radius(self):
        if self.nb_points == 1:
            return 0
        return self.get_deviation() * self.radius_factor

