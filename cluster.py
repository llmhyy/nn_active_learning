import pandas as pd
import numpy as np
import random
import math
import testing_function
# import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def get_point_from_cluster(cluster, num_to_be_added):

    # calculate mid point
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    centroid = []
    for i in range(dimension):
        tmp = 0
        for j in range(num_of_points):
            tmp += cluster[j][i]
        tmp = tmp/num_of_points
        centroid.append(tmp)

    print("Centroid: ", centroid)
    # calculate largest distance
    max_dist = 0
    for i in range(num_of_points):
        dist = 0
        for j in range(dimension):
            dist += math.sqrt((cluster[i][j]-centroid[j]) * (cluster[i][j]-centroid[j]))
        if dist > max_dist:
            max_dist = dist
    print("Max distance: ", max_dist)

    if max_dist == 0:
        max_dist = random.randint(10, 30)

    walk_dist = random.uniform(max_dist*1.5, max_dist*2)
    print("Walk distance: ", walk_dist)

    points_list = []
    diff_list = []
    for num in range(num_to_be_added):
        new_point = []
        diff_dist = 0
        for i in range(dimension):
            axis_dist = random.randint(10, 1000)
            new_point.append(axis_dist)
            diff_dist += axis_dist*axis_dist
        diff_list.append(diff_dist)
        points_list.append(new_point)

    for i in range(num_to_be_added):
        times = math.sqrt(diff_list[i]) / walk_dist
        points_list[i] = [p/times for p in points_list[i]]
    print("Points to be tested: ", points_list)
    return points_list

def get_clustering_points(X, label, formula):

    X = [[55,55],[65,56],[5,6],[4,6],[75,44],[7,2],[89,55],[68,86]]
    label = True
    formula = formula

    print(X)

    cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
    cluster.fit_predict(X)

    sep_clusters = {}
    for i in range(len(X)):
        sep_clusters[cluster.labels_[i]] = []

    for i in range(len(X)):
        sep_clusters[cluster.labels_[i]].append(X[i])

    print(sep_clusters)
    return_list = []
    for key in sep_clusters.keys():
        return_list += get_point_from_cluster(sep_clusters[key], 2)

    print(return_list)
    for point in return_list:
        flag = testing_function.test_label(point, formula)
        print(flag)
        if flag != label:
            return False
    return True
# plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
