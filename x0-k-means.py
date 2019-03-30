import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def distance_measure(x, mu, measure='Euclidean'):
    # x = np.atleast_2d(x)
   # mu = np.atleast_2d(mu)
    if measure == 'Euclidean':
        return np.sum((x[:,np.newaxis]-mu)**2, axis=-1)
    if measure == 'Manhattan':
        return np.sum(np.abs(x[:, np.newaxis]-mu), axis=-1)
    if measure == 'Cosine_Similarity':
        x_norm = np.sqrt(np.sum(x**2, keepdims=True, axis=-1))
        mu_norm = np.sqrt(np.sum(mu**2, keepdims=True, axis=-1))
        return np.dot(x, mu.T) / np.dot(x_norm, mu_norm.T)

"""
1) Implement the k-means clustering algorithm with Euclidean distance to cluster the instances
into k clusters.
"""

def my_kmeans(x,  k=4, measure='Euclidean'):
    """
   k: the number of centres
    
    """
    #  n:number of of sample, m: number of sample feature
    n, m = x.shape
    old_clusters_dict, clusters_dict = {}, {}
    # 1. Set k instances from the dataset randomly. (initial cluster centers)
    centres = np.random.permutation(np.arange(n))[:k]
    # get the centres
    mu = x[centres]
    maxiter_num = 100
    iter_num = 0
    while (clusters_dict == {} or old_clusters_dict != clusters_dict) and iter_num<maxiter_num:
        old_clusters_dict = clusters_dict
        clusters_dict = {}
        # caculate the distance, distance shape: n x k
        distance = distance_measure(x, mu,  measure)
        if measure == 'Cosine_Similarity':
            clusters = np.argmax(distance, axis=-1)
        else:
            clusters = np.argmin(distance, axis=-1)
        # 2. Assign all other instances to the closest cluster centre.
        for i in range(k):
            clusters_dict[i] = []
        for (idx, cluster)  in enumerate(clusters):
            clusters_dict[cluster].append(idx)
        # 3. Compute the mean of each cluster
        for cluster in range(k):
            mu[cluster] = np.mean(x[clusters_dict[cluster]], axis=0)
        iter_num += 1
    # 4. until convergence repeat between steps 2 and 3
    return clusters_dict



#tp, fn, fp, tn, tp_fp, tn_fn = 6 * [np.zeros(10, dtype='int')]
def get_PRF(x, xclass, measure='Euclidean'):
    tp = np.zeros(10)
    fn = np.zeros(10)
    fp   = np.zeros(10)
    tn   = np.zeros(10)
    tp_fp   = np.zeros(10)
    tn_fn  = np.zeros(10)
    num_class = 4
    for k in range(1, 11):
        cluster_class = np.zeros((k, num_class))
        xcluster = my_kmeans(x, k, measure=measure)

        for id_cluster in range(k):
            for item in xcluster[id_cluster]:
                id_class = xclass[item]
                cluster_class[id_cluster][id_class] += 1
       # print(cluster_class)
        same_cluster = np.sum(cluster_class, axis=-1)

        tp_fp[k-1] = np.sum((same_cluster - 1) * same_cluster / 2)
        tp[k-1] = np.sum((cluster_class - 1)* cluster_class / 2)
        fp[k-1] = tp_fp[k-1] - tp[k-1]

        for id_cluster1 in range(k):
            for id_cluster2 in range (id_cluster1, k):
                tn_fn[k-1] += same_cluster[id_cluster1] * same_cluster[id_cluster2]

        for id_class in range(num_class):
            for id_cluster in range(1,k):
                fn[k-1] += cluster_class[id_cluster][id_class]* cluster_class[id_cluster-1][id_class]
        tn[k-1] = tn_fn[k-1] - fn[k-1] 

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)
    return precision, recall, fscore
