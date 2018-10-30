import tensorflow as tf

import cluster as cl
import network_structure as ns
import util


def boundary_explore(data_set, label, model_path, label_tester, iterations):
    sample_point = data_set[0]
    dimension = len(sample_point)
    other_side_data = []

    with tf.Session() as sess:
        net = ns.NNStructure(dimension, 0.01)
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        for k in range(iterations):
            new_points = []
            centers, border_points_group, cluster_group = cl.cluster_points(data_set, 5, 3)
            for i in range(len(centers)):
                center = centers[i]
                border_points = border_points_group[i]

                step = util.calculate_std_dev(border_points)
                for border_point in border_points:
                    direction = border_point - center
                    new_point = util.move(border_point, direction, step)

                    if is_point_valid(new_point, net):
                        new_points.append(new_point)

            labels = label_tester.test_label(new_points)
            for i in range(len(labels)):
                if labels[i] != label:
                    other_side_data.append(new_points[i])

    return other_side_data


def is_point_valid(new_point, net):
    pass
