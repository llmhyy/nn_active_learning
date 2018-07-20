from __future__ import print_function

import math
import random

import numpy as np
import tensorflow as tf

import boundary_remaining as br
import formula
import testing_function
import util

step = 3


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


def is_training_data_balanced(length_0, length_1, balance_ratio_threshold):
    return (length_0 / length_1 > balance_ratio_threshold and length_0 / length_1 < 1) \
           or \
           (length_1 / length_0 > balance_ratio_threshold and length_1 / length_0 < 1)


def generate_accuracy(train_data_file, test_data_file, formu, category):
    print("=========MID_POINT===========")

    # Parameters
    learning_rate = 0.1
    training_epochs = 100

    balance_ratio_threshold = 0.7
    boundary_remaining_trial_iteration = 100

    to_be_appended_points_number = 10
    active_learning_iteration = 10
    threhold = 5
    test_set_X = []
    test_set_Y = []
    train_set_X = []
    train_set_Y = []

    util.preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y, train_data_file, test_data_file, read_next=True)
    # Network Parameters
    n_hidden_1 = 10  # 1st layer number of neurons
    n_hidden_2 = 10  # 2nd layer number of neurons
    n_input = len(train_set_X[0])  # MNIST data input (img shape: 28*28)
    n_classes = 1  # MNIST total classes (0-9 digits)

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)

    train_acc_list = []
    test_acc_list = []
    result = []

    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0)),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    logits = util.multilayer_perceptron(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Initializing the variables
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    new_grads = tf.gradients(logits, X)
    y = None

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))
        to_be_appended_points_number = 10

        #TODO add a to_be_randomed_points_number = 10

        with tf.Session() as sess:
            sess.run(init)
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            print(len(label_0), len(label_1))

            if (len(label_1) == 0 or len(label_0) == 0):
                raise Exception("Cannot be classified")

            distance_list, point_pair_list = filter_distant_point_pair(label_0, label_1, threhold)

            util.quickSort(distance_list)

            append_mid_points(distance_list, formu, point_pair_list, to_be_appended_points_number,
                              train_set_X, train_set_Y)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

            if (not is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
                br.apply_boundary_remaining(sess, new_grads, X, Y, length_0, length_1, logits, formu, train_set_X, train_set_Y)

            for epoch in range(training_epochs):
                _, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X, Y: train_set_Y})

            train_y = sess.run(logits, feed_dict={X: train_set_X})
            test_y = sess.run(logits, feed_dict={X: test_set_X})

            print("new train size after mid point", len(train_set_X), len(train_set_Y))
            train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            test_acc = util.calculate_accuracy(test_y, test_set_Y, False)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            predicted = tf.cast(logits > 0.5, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={X:x}), train_set_X, train_set_Y)

    result.append(train_acc_list)
    result.append(test_acc_list)
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
                        train_set_Y.append([0])
                else:
                    if (middle_point not in train_set_X):
                        train_set_X.append(middle_point)
                        train_set_Y.append([1])
