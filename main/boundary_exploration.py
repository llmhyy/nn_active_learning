import os
import random

import numpy as np
import tensorflow as tf

from main import util, cluster as cl, network_structure as ns


class BoundaryExplorer:
    def __init__(self, data_set_info, data_set, model_folder, model_file, label_tester,
                 iterations):
        self.data_set_info = data_set_info
        self.data_set = data_set
        self.model_folder = model_folder
        self.model_file = model_file
        self.label_tester = label_tester
        self.iterations = iterations

    def boundary_explore(self):
        sample_point = self.data_set[0]
        other_side_data = []

        sess = None
        net = None

        if self.model_folder is not None and os.path.exists(self.model_folder):
            model_path = os.path.join(self.model_folder, self.model_file)
            if os.path.exists(model_path + ".meta"):
                tf.reset_default_graph()
                net = ns.NNStructure(len(sample_point), 0.01)

                sess = tf.Session()
                saver = tf.train.Saver()
                saver.restore(sess, model_path)

                # net.print_parameters(sess)
                # p = sess.run(net.probability, feed_dict={net.X: [[325, -302]]})
                pass

        for k in range(self.iterations):
            print(k + 1, "th iteration")
            new_points = []
            centers, _, cluster_group = cl.cluster_points(self.data_set, 1, 5)
            util.plot_clustering_result(cluster_group, -1000, 1000, k + 1)
            for i in range(len(centers)):
                center = centers[i]
                cluster = cluster_group[i]

                distance_from_farthest_border = cl.calculate_distance_from_farthest_border(cluster)
                border_points = cl.random_border_points(center, distance_from_farthest_border, 5)

                std_dev = util.calculate_std_dev(border_points)
                step = random.uniform(0, std_dev)
                # step = 5
                for border_point in border_points:
                    direction = (np.array(border_point) - np.array(center)).tolist()
                    new_point = util.move(border_point, direction, step)

                    if self.is_point_inside_boundary(sess, new_point, net):
                        is_too_close = self.check_closeness(new_point, cluster_group)
                        if not is_too_close:
                            new_points.append(new_point)

        print("added new points", len(new_points))
        print(new_points)
        if len(new_points) > 0:
            new_points = util.convert_with_data_type_and_mask(new_points, self.data_set_info, self.label_tester)
            labels = self.label_tester.test_label(new_points)
            for i in range(len(labels)):
                if labels[i] == 1:
                    other_side_data.append(new_points[i])
                else:
                    self.data_set.append(new_points[i])

        if sess is not None:
            sess.close()

        print(other_side_data)
        return other_side_data

    def is_point_inside_boundary(self, sess, new_point, net):
        if sess is None:
            return True
        prob = sess.run(net.probability, feed_dict={net.X: [new_point]})
        prediction = 0
        if prob[0] >= 0.5:
            prediction = 1
        return prediction == 1

    def check_closeness(self, new_point, cluster_group):
        for cluster in cluster_group:
            center = cl.calculate_center(cluster)
            boundary_distance = cl.calculate_distance_from_farthest_border(cluster)
            distance = util.calculate_distance(center, new_point)
            if distance < boundary_distance:
                return True

        return False
