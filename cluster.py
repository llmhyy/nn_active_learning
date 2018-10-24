import math
import random
import util

# import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

import testing_function


def calculate_farthest_point_distance(cluster):
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    center = calculate_center(cluster)
    largest = 0
    for i in range(num_of_points):
        tmp = 0
        for j in range(dimension):
            tmp += (cluster[i][j] - center[j]) * (cluster[i][j] - center[j])
        tmp = math.sqrt(tmp)
        if tmp > largest:
            largest = tmp

    return largest


def calculate_center(cluster):
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    centroid = []
    for i in range(dimension):
        tmp = 0
        for j in range(num_of_points):
            tmp += cluster[j][i]
        tmp = tmp / num_of_points
        centroid.append(tmp)

    return centroid


def get_point_from_cluster(cluster, num_to_be_added):
    # calculate mid point
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    centroid = calculate_center(cluster)

    # print("Centroid: ", centroid)
    # calculate largest distance
    max_dist = 0
    for i in range(num_of_points):
        dist = 0
        for j in range(dimension):
            dist += math.sqrt((cluster[i][j] - centroid[j]) * (cluster[i][j] - centroid[j]))
        if dist > max_dist:
            max_dist = dist
    # print("Max distance: ", max_dist)

    if max_dist == 0:
        max_dist = random.randint(10, 30)

    walk_dist = random.uniform(max_dist * 1.5, max_dist * 2)
    # print("Walk distance: ", walk_dist)

    points_list = []
    diff_list = []
    for num in range(num_to_be_added):
        new_point = []
        diff_dist = 0
        for i in range(dimension):
            axis_dist = random.randint(10, 1000)
            new_point.append(axis_dist)
            diff_dist += axis_dist * axis_dist
        diff_list.append(diff_dist)
        points_list.append(new_point)

    for i in range(num_to_be_added):
        times = math.sqrt(diff_list[i]) / walk_dist
        points_list[i] = [p / times for p in points_list[i]]
    # print("Points to be tested: ", points_list)
    return points_list


def is_clustering_valid(clusters):
    if len(clusters) == 1:
        return True

    centers = []
    farthest_point_distance_list = []
    for cluster in clusters:
        dimension = len(clusters[cluster][0])
        num_of_points = len(clusters[cluster])

        center = calculate_center(clusters[cluster])
        centers.append(center)

        farthest_point_distance = calculate_farthest_point_distance(clusters[cluster])
        farthest_point_distance_list.append(farthest_point_distance)

    centers.append(centers[0])
    farthest_point_distance_list.append(farthest_point_distance_list[0])

    for j in range(len(clusters)):
        distance = 0
        for i in range(len(clusters[cluster][0])):
            distance += (centers[j][i] - centers[j + 1][i])**2
        distance = math.sqrt(distance)

        if farthest_point_distance_list[j] * 3 > distance or farthest_point_distance_list[j + 1] * 3 > distance:
            return False

    return True


def get_clustering_points(X, label, formula):
    # X = [[55,55],[65,56],[5,6],[4,6],[75,44],[7,2],[89,55],[68,86]]
    label = True
    formula = formula
    num_cluster = 3

    # print(X)
    while True:

        cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='ward')
        cluster.fit_predict(X)

        sep_clusters = {}
        for i in range(len(X)):
            sep_clusters[cluster.labels_[i]] = []

        for i in range(len(X)):
            sep_clusters[cluster.labels_[i]].append(X[i])

        if is_clustering_valid(sep_clusters):
            break
        else:
            num_cluster -= 1

    print("Final number of clusters: ", num_cluster)
    # print(sep_clusters)
    return_list = []
    for key in sep_clusters.keys():
        return_list += get_point_from_cluster(sep_clusters[key], 2)

    # print(return_list)
    for point in return_list:
        flag = testing_function.test_label(point, formula)
        if flag != label:
            print("Found different label")
            return False, return_list

    X = X + return_list
    return True, X


# plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
def cluster_points(X, n):
    # X = [[55,55],[65,56],[5,6],[4,6],[75,44],[7,2],[89,55],[68,86]]
    num_cluster = 10

    # print(X)
    while True:

        cluster = AgglomerativeClustering(n_clusters=num_cluster, affinity='euclidean', linkage='average')
        cluster.fit_predict(X)

        sep_clusters = {}
        for i in range(len(X)):
            sep_clusters[cluster.labels_[i]] = []

        for i in range(len(X)):
            sep_clusters[cluster.labels_[i]].append(X[i])

        util.plot_clustering_result(sep_clusters, -1000, 1000, 1)

        if is_clustering_valid(sep_clusters):
            break
        else:
            num_cluster -= 1

    print("Final number of clusters: ", num_cluster)
    print(sep_clusters)
    centers = []
    n_farthest_distances = []
    for key in sep_clusters.keys():
        cluster = sep_clusters[key]
        center = calculate_center(cluster)
        centers.append(center)
        farthest_distance_list = calculate_n_farthest_distances(cluster, center, n)
        n_farthest_distances.append(farthest_distance_list)

    return centers, n_farthest_distances


def calculate_n_farthest_distances(cluster, center, n):
    distance = []
    result = []
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    for i in range(num_of_points):
        tmp = 0
        for j in range(dimension):
            tmp += (cluster[i][j] - center[j]) * (cluster[i][j] - center[j])
        tmp = math.sqrt(tmp)
        distance.append(tmp)
    sorted_dist = sorted(distance)

    threshold = sorted_dist[-n]
    for i in range(len(distance)):
        if distance[i] >= threshold:
            result.append(cluster[i])

    return result
