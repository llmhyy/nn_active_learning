import os
import random

import numpy as np
import tensorflow as tf

from main import util, cluster as cl, network_structure as ns


def boundary_explore(data_set, model_folder, model_file, label_tester, info_checker,
                     iterations):
    sample_point = data_set[0]
    other_side_data = []

    sess = None
    net = None

    if model_folder is not None and os.path.exists(model_folder):
        model_path = os.path.join(model_folder, model_file)
        if os.path.exists(model_path + ".meta"):
            tf.reset_default_graph()
            net = ns.NNStructure(len(sample_point), 0.01)

            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            # net.print_parameters(sess)
            # p = sess.run(net.probability, feed_dict={net.X: [[325, -302]]})
            pass

    for k in range(iterations):
        print(k + 1, "th iteration")
        new_points = []
        centers, _, cluster_group = cl.cluster_points(data_set, 1, 5)
        util.plot_clustering_result(cluster_group, -1000, 1000, k + 1)
        for i in range(len(centers)):
            center = centers[i]
            cluster = cluster_group[i]

            distance_from_farthest_border = cl.calculate_distance_from_farthest_border(cluster)
            border_points = cl.random_border_points(center, distance_from_farthest_border, 5)

            std_dev = util.calculate_std_dev(border_points)
            step = random.uniform(0, std_dev)
            # step = 5
            new_point_list = []
            for border_point in border_points:
                direction = (np.array(border_point) - np.array(center)).tolist()
                new_point = util.move(border_point, direction, step)
                # TODO reset new point with data type
                new_point_list.append(new_point)

            new_point_list_info = info_checker.check_info(new_point_list)
            new_point_list = util.convert_with_mask(new_point_list, new_point_list_info)

            for new_point in new_point_list:
                if is_point_inside_boundary(sess, new_point, net):
                    is_too_close = check_closeness(new_point, cluster_group)
                    if not is_too_close:
                        new_points.append(new_point)

    print("added new points", len(new_points))
    print(new_points)
    if len(new_points) > 0:
        labels = label_tester.test_label(new_points)
        for i in range(len(labels)):
            if labels[i] == 1:
                other_side_data.append(new_points[i])
            else:
                data_set.append(new_points[i])

    if sess is not None:
        sess.close()

    print(other_side_data)
    return other_side_data


def is_point_inside_boundary(sess, new_point, net):
    if sess is None:
        return True
    prob = sess.run(net.probability, feed_dict={net.X: [new_point]})
    prediction = 0
    if prob[0] >= 0.5:
        prediction = 1
    return prediction == 1


def check_closeness(new_point, cluster_group):
    for cluster in cluster_group:
        center = cl.calculate_center(cluster)
        boundary_distance = cl.calculate_distance_from_farthest_border(cluster)
        distance = util.calculate_distance(center, new_point)
        if distance < boundary_distance:
            return True

    return False
