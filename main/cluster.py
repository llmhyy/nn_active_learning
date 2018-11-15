import math

import random
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from main import angle, util


def calculate_radius(cluster):
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    center = calculate_center(cluster)
    total_sum = 0
    for i in range(num_of_points):
        tmp = 0
        for j in range(dimension):
            tmp += (cluster[i][j] - center[j]) ** 2
        tmp = math.sqrt(tmp)
        total_sum = total_sum + tmp

    average = total_sum / num_of_points;

    return average


def calculate_distance_from_farthest_border(cluster):
    dimension = len(cluster[0])
    num_of_points = len(cluster)
    center = calculate_center(cluster)
    largest_distance = 0
    for i in range(num_of_points):
        tmp = 0
        for j in range(dimension):
            tmp += (cluster[i][j] - center[j]) ** 2
        distance = math.sqrt(tmp)
        if distance > largest_distance:
            largest_distance = distance

    if largest_distance == 0:
        largest_distance = np.random.uniform(0, 10)

    return largest_distance


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


def is_clustering_valid(clusters, cluster_distance_threshold):
    if len(clusters) == 1:
        return True

    centers = []
    radius_list = []
    for key in clusters:
        dimension = len(clusters[key][0])
        num_of_points = len(clusters[key])

        center = calculate_center(clusters[key])
        centers.append(center)

        radius = calculate_radius(clusters[key])
        radius_list.append(radius)

    centers.append(centers[0])
    radius_list.append(radius_list[0])

    for j in range(len(clusters)):
        center_distance = 0
        for i in range(len(clusters[key][0])):
            center_distance += (centers[j][i] - centers[j + 1][i]) ** 2
        center_distance = math.sqrt(center_distance)

        if radius_list[j] * cluster_distance_threshold > center_distance \
                or radius_list[j + 1] * cluster_distance_threshold > center_distance:
            return False

    return True


# plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
def cluster_points(data_set, border_point_number, maximum_num_cluster):
    # X = [[55,55],[65,56],[5,6],[4,6],[75,44],[7,2],[89,55],[68,86]]
    # for each two cluster center, their threshold*radius should be larger than the center distance
    if len(data_set) == 1:
        sep_clusters = {0: data_set}
    else:
        cluster_distance_threshold = 2
        if maximum_num_cluster > len(data_set):
            maximum_num_cluster = len(data_set)

        while True:
            cluster = AgglomerativeClustering(n_clusters=maximum_num_cluster, affinity='euclidean', linkage='average')
            cluster.fit_predict(data_set)

            sep_clusters = {}
            for i in range(len(data_set)):
                sep_clusters[cluster.labels_[i]] = []

            for i in range(len(data_set)):
                sep_clusters[cluster.labels_[i]].append(data_set[i])

            if is_clustering_valid(sep_clusters, cluster_distance_threshold):
                break
            else:
                maximum_num_cluster -= 1

    # print("Final number of clusters: ", maximum_num_cluster)
    # print(sep_clusters)
    centers = []
    border_points_group = []
    cluster_group = []
    for key in sep_clusters.keys():
        cluster = sep_clusters[key]
        cluster_group.append(cluster)
        center = calculate_center(cluster)
        centers.append(center)
        border_points = calculate_n_border_points(cluster, center, border_point_number)
        border_points_group.append(border_points)

    return centers, border_points_group, cluster_group


def calculate_n_border_points(cluster, center, n):
    angle_list = []
    num_of_points = len(cluster)

    if n >= num_of_points:
        return cluster

    if n == 1:
        return [random.choice(cluster)]

    for i in range(num_of_points):
        for j in range(i+1, num_of_points):
            point1 = cluster[i]
            point2 = cluster[j]
            single_angle = angle.Angle(center, point1, point2)
            angle_list.append(single_angle)

        angle_list.sort(key=lambda x: x.angle, reverse=True)

    border_points = []
    for single_angle in angle_list:
        append_points(border_points, single_angle, angle_list)
        if len(border_points) > n:
            break;

    extend_to_border(center, border_points, angle_list)

    return border_points


def extend_to_border(center, border_points, angle_list):
    for i in range(len(border_points)):
        border_point = border_points[i]
        bench_distance = util.calculate_distance(center, border_point)

        best_candidate = []
        distance = -1
        for single_angle in angle_list:
            other_point = single_angle.get_other_point(border_point)
            if other_point is not None and single_angle.angle < math.pi/4:
                other_distance = util.calculate_distance(center, other_point)
                if other_distance > distance:
                    distance = other_distance
                    best_candidate = other_point

        if distance > bench_distance:
            border_points[i] = best_candidate


def append_points(border_points, single_angle, angle_list):
    point1 = single_angle.point1
    point2 = single_angle.point2
    is_close = is_close_angle(border_points, point1, angle_list)
    if not is_close:
        border_points.append(point1)

    is_close = is_close_angle(border_points, point2, angle_list)
    if not is_close:
        border_points.append(point2)


def is_close_angle(border_points, point, angle_list):
    if len(border_points) == 0:
        return False

    if point in border_points:
        return True
    else:
        for border_point in border_points:
            for single_angle in angle_list:
                is_find = single_angle.find(border_point, point)
                if is_find:
                    if single_angle.angle < math.pi/4:
                        return True
        return False

def random_border_points(center, radius, border_point_number):
    border_points = []

    while len(border_points) < border_point_number:
        border_point = []
        value = 0
        for j in range(len(center) - 1):
            x = center[j]
            y = np.random.uniform(x - radius/math.sqrt(len(center)), x + radius/math.sqrt(len(center)))
            border_point.append(y)

            value += (x - y) ** 2

        diff = radius ** 2 - value
        if diff > 0:
            last_dim = math.sqrt(diff)
            r = np.random.uniform(0, 1)
            if r < 0.5:
                last_dim = -last_dim

            border_point.append(last_dim)
            border_points.append(border_point)

    return border_points
