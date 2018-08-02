from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

import boundary_remaining as br
import testing_function
import util
import network_structure as ns


def filter_distant_point_pair(label_0, label_1, threshold):
    distance_list = []
    point_pair_list = {}

    for m in label_0:
        for n in label_1:
            distance = util.calculate_distance(m, n)
            if (distance > threshold):
                key = ()
                for h in range(len(m)):
                    tmp_key = (m[h], n[h])
                    key = key + tmp_key
                value = distance
                if (not point_pair_list):
                    point_pair_list[key] = value
                elif (not (key in point_pair_list.keys())):
                    point_pair_list[key] = value
                distance_list.append(distance)

    return distance_list, point_pair_list


def generate_accuracy(train_data_file, test_data_file, formu, category, learning_rate, training_epochs, lower_bound, upper_bound):
    print("=========MID_POINT===========")
    balance_ratio_threshold = 0.7
    boundary_remaining_trial_iteration = 100

    to_be_appended_points_number = 6
    to_be_appended_boundary_remaining_points_number = 6
    to_be_appended_random_points_number = 6
    active_learning_iteration = 5
    threshold = 5


    train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_data_file, test_data_file, read_next=True)

    net_stru = ns.NNStructure(train_set_X[0], learning_rate)


    train_acc_list = []
    test_acc_list = []
    result = []

    new_grads = tf.gradients(net_stru.logits, net_stru.X)
    y = None

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))

        # TODO add a to_be_randomed_points_number = 10

        with tf.Session() as sess:
            sess.run(net_stru.init)
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print(length_0, length_1)
            if (length_0 == 0 or length_1 == 0):
                raise Exception("Cannot be classified")

            if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
                br.apply_boundary_remaining(sess, new_grads, net_stru.X, net_stru.Y, length_0, length_1, net_stru.logits, formu, train_set_X,
                                            train_set_Y, to_be_appended_boundary_remaining_points_number)

            for epoch in range(training_epochs):
                _, c = sess.run([net_stru.train_op, net_stru.loss_op], feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})

            train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
            test_y = sess.run(net_stru.logits, feed_dict={net_stru.X: test_set_X})

            train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            test_acc = util.calculate_accuracy(test_y, test_set_Y, False)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            # predicted = tf.cast(net_stru.logits > 0.5, dtype=tf.float32)
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X, train_set_Y, i)

            distance_list, point_pair_list = filter_distant_point_pair(label_0, label_1, threshold)
            util.quickSort(distance_list)
            append_mid_points(distance_list, formu, point_pair_list, to_be_appended_points_number,
                              train_set_X, train_set_Y)
            print("new train size after mid point", len(train_set_X), len(train_set_Y))
            train_set_X, train_set_Y = util.append_random_points(formu, train_set_X, train_set_Y,
                                                                 to_be_appended_random_points_number, lower_bound, upper_bound)
            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

    result.append(train_acc_list)
    result.append(test_acc_list)
    tf.reset_default_graph()
    return result


def append_mid_points(distance_list, formu, point_pair_list, to_be_appended_points_number,
                      train_set_X, train_set_Y):
    selected_distance_list = []
    length = len(distance_list)
    index1 = int(length / 3)
    index2 = int(length / 3 * 2)
    pointer = 0
    for p in range(3):
        if (pointer < index1):
            num = int(to_be_appended_points_number * 0.6)
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)
            pointer = index1
        elif (pointer < index2):
            num = int(to_be_appended_points_number * 0.3)
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)
            #
            pointer = index2
        else:
            num = int(to_be_appended_points_number * 0.1)
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)

    for distance in selected_distance_list:
        for point_key, dis_value in point_pair_list.items():
            if (distance == dis_value):
                point_0 = []
                point_1 = []
                for b in range(len(point_key)):
                    if (b % 2 == 0):
                        point_0.append(point_key[b])
                    else:
                        point_1.append(point_key[b])
                middle_point = []

                for b in range(len(point_0)):
                    middle_point.append((point_0[b] + point_1[b]) / 2.0)

                label = testing_function.test_label(middle_point, formu)

                if (label):
                    if (middle_point not in train_set_X):
                        train_set_X.append(middle_point)
                        train_set_Y.append([1])
                else:
                    if (middle_point not in train_set_X):
                        train_set_X.append(middle_point)
                        train_set_Y.append([0])
