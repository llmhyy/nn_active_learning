import os

import tensorflow as tf
import numpy as np
import random

from main import network_structure as ns, gradient_util, util


class BoundaryRemainer:
    def __init__(self, positive_points_info, positive_points, model_folder, model_file,
                 selected_point_number):
        self.positive_points_info = positive_points_info
        self.positive_points = positive_points
        self.model_folder = model_folder
        self.model_file = model_file
        self.selected_point_number = selected_point_number

    def search_remaining_boundary_points(self):
        new_point_list = []
        if self.model_folder is None or not os.path.exists(self.model_folder):
            return new_point_list
        model_path = os.path.join(self.model_folder, self.model_file)
        if not os.path.exists(model_path + ".meta"):
            return new_point_list

        input_dimension = len(self.positive_points[0])
        tf.reset_default_graph()
        with tf.Session() as sess:
            net = ns.NNStructure(input_dimension, 0.01)

            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, model_path)

            gradient_tensor = tf.gradients(net.probability, net.X)
            gradients = sess.run(gradient_tensor, feed_dict={net.X: self.positive_points})
            gradients = gradient_util.confirm_gradient_direction(sess, self.positive_points, net, gradients)

            std_dev = util.calculate_std_dev(self.positive_points)
            for k in range(len(self.positive_points)):
                point = self.positive_points[k]
                gradient = gradients[k]
                # step = 10
                normal_vector = gradient_util.calculate_opposite_vector(gradient)
                normal_vector_length = util.calculate_vector_size(normal_vector)

                for j in range(10):
                    step = np.random.uniform(0, std_dev)
                    new_point = []
                    for i in range(len(normal_vector)):
                        dimension_value = point[i]
                        direction = normal_vector[i]
                        new_dimension = dimension_value + step*direction/normal_vector_length
                        new_point.append(new_dimension)

                    p = sess.run(net.probability, feed_dict={net.X: [new_point]})[0]
                    if p > 0.5:
                        break;

                if p > 0.5:
                    new_point_list.append(new_point)

        if len(new_point_list) > self.selected_point_number:
            random.shuffle(new_point_list)
            new_point_list = new_point_list[0: self.selected_point_number]

        return new_point_list


class Point:
    def __init__(self, point, probability):
        self.point = point,
        self.probability = probability

