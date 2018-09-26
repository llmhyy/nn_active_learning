from __future__ import print_function

import random

import tensorflow as tf

import boundary_remaining as br
import network_structure as ns
import testing_function
import util
import data_pair
import operator


def partition_data(label_0, label_1, parts_num):
    result_X = []
    result_Y = []
    tmpx0 = []
    tmpx1 = []
    tmpy0 = []
    tmpy1 = []

    min_size = min(len(label_0), len(label_1))
    max_size = max(len(label_0), len(label_1))

    if max_size < 2 * min_size:
        tmpx0 = label_0
        tmpy0 = [[0] for j in tmpx0]
        tmpx1 = label_1
        tmpy1 = [[1] for j in tmpx1]
        tmpX = tmpx0 + tmpx1
        tmpY = tmpy0 + tmpy1
        result_X.append(tmpX)
        result_Y.append(tmpY)
    else:
        if len(label_1) < len(label_0):
            tmpx1 = label_1
            tmpy1 = [[1] for j in tmpx1]
        else:
            tmpx0 = label_0
            tmpy0 = [[0] for j in tmpx0]

        for i in range(parts_num):
            if len(label_1) < len(label_0):
                tmpx0 = random.sample(label_0, len(label_1))
                tmpy0 = [[0] for j in tmpx0]
            else:
                tmpx1 = random.sample(label_1, len(label_0))
                tmpy1 = [[1] for j in tmpx1]
            tmpX = tmpx0 + tmpx1
            tmpY = tmpy0 + tmpy1

            result_X.append(tmpX)
            result_Y.append(tmpY)

    return result_X, result_Y


def filter_distant_point_pair(label_0, label_1, threshold):
    """
    find a list of data points whose distance is over the threshold

    :param label_0:
    :param label_1:
    :param threshold:
    :return:
    """

    pair_list = [];

    for m in label_0:
        for n in label_1:
            distance = util.calculate_distance(m, n)
            if (distance > threshold):

                pair = data_pair.DataPair(m, n, distance)
                pair_list.append(pair)

    return pair_list


def generate_accuracy(inputX, inputY, train_data_file, test_data_file, formu, category, learning_rate, training_epochs,
                      lower_bound, upper_bound, use_bagging, type, name_list, mock):
    print("=========MID_POINT===========")
    balance_ratio_threshold = 0.7
    boundary_remaining_trial_iteration = 100

    to_be_appended_points_number = 6
    to_be_appended_boundary_remaining_points_number = 3
    # to_be_appended_random_points_number = 3
    active_learning_iteration = 5
    threshold = 100
    save_path = "model_saved/gradient_model"
    train_set_X = []
    train_set_Y = []
    test_set_X = []
    test_set_Y = []
    if mock == True:
        train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_data_file, test_data_file,
                                                                           read_next=True)
    else:
        train_set_X = inputX
        train_set_Y = inputY

    net_stru = ns.NNStructure(train_set_X[0], learning_rate)

    ## save weights and bias
    # saver = tf.train.Saver()

    train_acc_list = []
    test_acc_list = []
    train_acc_max = 0

    result = []

    new_grads = tf.gradients(net_stru.logits, net_stru.X)
    y = None

    predicted = tf.cast(net_stru.logits > 0, dtype=tf.float32)

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
            # print(label_1)
            initial_point = len(label_0)
            if (length_0 == 0 or length_1 == 0):
                raise Exception("Cannot be classified")

            if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
                br.apply_boundary_remaining(sess, new_grads, net_stru.X, net_stru.Y, length_0, length_1,
                                            net_stru.logits, formu, train_set_X,
                                            train_set_Y, to_be_appended_boundary_remaining_points_number, type,
                                            name_list, mock)
                util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
                                            train_set_Y, lower_bound, upper_bound, 10 + i)
                print("new training size after boundary remaining", "X: ", len(train_set_X))

            # print(train_set_X)
            # print(train_set_Y)
            smaller_set_size = min(len(label_0), len(label_1))
            larger_set_size = max(len(label_0), len(label_1))
            parts_num = int(larger_set_size / smaller_set_size)
            parts_num = 2

            all_data_X, all_data_Y = partition_data(label_0, label_1, parts_num)
            # print(all_data_X, all_data_Y)

            all_weights_dict = []
            all_biases_dict = []

            if use_bagging:
                for parts in range(parts_num):
                    best_accuracy = 0
                    sess.run(net_stru.init)
                    for epoch in range(training_epochs):
                        _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                        feed_dict={net_stru.X: all_data_X[parts], net_stru.Y: all_data_Y[parts]})
                        train_y = sess.run(net_stru.logits, feed_dict={
                            net_stru.X: train_set_X})
                        train_acc = util.calculate_accuracy(
                            train_y, train_set_Y, False)

                    # predicted = tf.cast(net_stru.logits > 0.5, dtype=tf.float32)
                    # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}),
                    #                             train_set_X, train_set_Y,
                    #                             lower_bound, upper_bound, -1)

                    weights_dict = sess.run(net_stru.weights)
                    bias_dict = sess.run(net_stru.biases)

                    all_weights_dict.append(weights_dict)
                    all_biases_dict.append(bias_dict)

                aggregated_network = ns.AggregateNNStructure(
                    train_set_X[0], learning_rate, all_weights_dict, all_biases_dict)
                sess.run(aggregated_network.init)

                train_y = sess.run(aggregated_network.logits, feed_dict={
                    aggregated_network.X: train_set_X})
                train_acc = util.calculate_accuracy(train_y, train_set_Y, print_data_details=False)
                print("Bagging performance", "train_acc", train_acc)
                # with tf.Session as session:
                #     session.run(net_stru_.init)
            else:
                for epoch in range(training_epochs):
                    _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                    feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})
                aggregated_network = net_stru

                train_y = sess.run(aggregated_network.logits, feed_dict={
                    aggregated_network.X: train_set_X})
                train_acc = util.calculate_accuracy(
                    train_y, train_set_Y, print_data_details=False)
                print("train_acc", train_acc)
                # if train_acc > best_accuracy:
                #     best_accuracy = train_acc
                # print("best_accuracy ", best_accuracy)
                # saver.save(sess, './models/benchmark.ckpt')

            # saver.restore(sess, "./models/benchmark.ckpt")
            # sess.run(net_stru_.init)

            ## save model
            # if len(train_acc_list) == 0:
            #     # net_stru_ = ns.NNStructure_save(train_set_X[0], learning_rate)
            #     saver.save(sess, save_path)
            #     print("Model saved")
            #     train_acc_max = train_acc
            # else:
            #     if train_acc >= train_acc_max:
            #         print("Got better result")
            #         # net_stru_ = ns.NNStructure_save(train_set_X[0], learning_rate)
            #         saver.save(sess, save_path)
            #         train_acc_max = train_acc
            #     else:
            #         print("not a better result")

            train_acc_list.append(train_acc)
            threshold = util.calculate_std_dev(train_set_X)
            predicted = tf.cast(net_stru.logits > 0, dtype=tf.float32)

            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru_.X: x}), train_set_X,train_set_Y, lower_bound, upper_bound, i)
            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            pair_list = filter_distant_point_pair(label_0, label_1, threshold)
            sorted(pair_list, key=operator.attrgetter('distance'))
            pass
            # print ("sorted list: ",distance_list)
            append_mid_points(pair_list, formu, to_be_appended_points_number,
                              train_set_X, train_set_Y, type, name_list, mock)

            print("new train size after mid point", len(train_set_X), len(train_set_Y))
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, lower_bound, upper_bound, 20 + i)

            # train_set_X, train_set_Y = util.append_random_points(formu, train_set_X, train_set_Y,
            #                                                      to_be_appended_random_points_number, lower_bound, upper_bound,type,name_list,mock)
            # print ("size after random points:",len(train_set_X))
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, lower_bound,upper_bound,30+i)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)
            for index in range(len(train_set_X)):
                print(train_set_X[index], train_set_Y[index])
            # print("weights trained:", sess.run(net_stru.weights))

            # for m in range(initial_point,len(label_0)):
            #     print (label_0[m])
            # print ("weights after :",sess.run(net_stru.weights))

    print("$TRAINING_FINISH")
    # with tf.Session() as sess:
    #     saver.restore(sess, save_path)
    #     # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
    #     #                             train_set_Y, lower_bound,upper_bound,-1)
    #     print("weights final:", sess.run(net_stru.weights))
    #     train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
    #     # test_y = sess.run(net_stru.logits, feed_dict={net_stru.X: test_set_X})
    #
    #     train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
    #     # test_acc = util.calculate_accuracy(test_y, test_set_Y, False)
    #     train_acc_list.append(train_acc)
    #     # test_acc_list.append(test_acc)
    result.append(train_acc_list)
    result.append(test_acc_list)
    print("Result", result)
    tf.reset_default_graph()
    return result


def append_mid_points(pair_list, formu, to_be_appended_points_number,
                      train_set_X, train_set_Y, type, name_list, mock):
    selected_distance_list = []
    length = len(pair_list)
    index1 = int(length / 3)
    index2 = int(length / 3 * 2)




    # pointer = 0
    # for p in range(3):
    #     if (pointer < index1):
    #         num = to_be_appended_points_number * 0.4
    #
    #         num = int(round(num))
    #
    #         if (num < 1):
    #             num = 1
    #
    #         util.add_distance_values(num, distance_list, selected_distance_list, pointer)
    #         pointer = index1
    #     elif (pointer < index2):
    #         num = to_be_appended_points_number * 0.3
    #         num = int(round(num))
    #
    #         if (num < 1):
    #             num = 1
    #
    #         util.add_distance_values(num, distance_list, selected_distance_list, pointer)
    #         #
    #         pointer = index2
    #     else:
    #         num = to_be_appended_points_number * 0.3
    #         num = int(round(num))
    #
    #         if (num < 1):
    #             num = 1
    #
    #         util.add_distance_values(num, distance_list, selected_distance_list, pointer)
    # print("all list:", distance_list)
    # print("selected list:", selected_distance_list)
    #
    # for distance in selected_distance_list:
    #     for point_key, dis_value in point_pair_list.items():
    #         if (distance == dis_value):
    #             point_0 = []
    #             point_1 = []
    #             for b in range(len(point_key)):
    #                 if (b % 2 == 0):
    #                     point_0.append(point_key[b])
    #                 else:
    #                     point_1.append(point_key[b])
    #             middle_point = []
    #             input_points = []
    #             for b in range(len(point_0)):
    #                 middle_point.append((point_0[b] + point_1[b]) / 2.0)
    #             input_points.append(middle_point)
    #             for point in input_points:
    #                 for index in range(len(point)):
    #                     if type == "INTEGER":
    #                         point[index] = int(round(point[index]))
    #
    #             result = testing_function.test_label(input_points, formu, type, name_list, mock)
    #             label = None
    #             if result[0] == 0:
    #                 label = False
    #             else:
    #                 label = True
    #
    #             if (label):
    #                 if (middle_point not in train_set_X):
    #                     train_set_X.append(middle_point)
    #                     train_set_Y.append([1])
    #                     print("from:", point_0, point_1, "middle point:", middle_point, label)
    #             else:
    #                 if (middle_point not in train_set_X):
    #                     train_set_X.append(middle_point)
    #                     train_set_Y.append([0])
    #                     print("from:", point_0, point_1, "middle point:", middle_point, label)
