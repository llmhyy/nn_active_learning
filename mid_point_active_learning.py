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

    to_be_appended_points_number = 2
    to_be_appended_boundary_remaining_points_number = 1
    to_be_appended_random_points_number = 1

    active_learning_iteration = 5
    threshold = 100
    save_path="model_saved/mid_point_model"

    train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_data_file, test_data_file, read_next=True)

    net_stru = ns.NNStructure(train_set_X[0], learning_rate)

    ## save weights and bias
    saver=tf.train.Saver()

    train_acc_list = []
    test_acc_list = []
    train_acc_max=0

    result = []

    new_grads = tf.gradients(net_stru.logits, net_stru.X)
    y = None

    predicted = tf.cast(net_stru.logits > 0, dtype=tf.float32)

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))

        # TODO add a to_be_randomed_points_number = 10

        with tf.Session() as sess:
            # if len(train_acc_list)==0:
            #
            #     sess.run(net_stru.init)
            # if len(train_acc_list)!=0:
            #     saver.restore(sess,save_path)
            sess.run(net_stru.init)
            print ("weights after restore:",sess.run(net_stru.weights))
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print(length_0, length_1)
            print (label_1)
            initial_point = len(label_0)
            if (length_0 == 0 or length_1 == 0):
                raise Exception("Cannot be classified")

            if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
                br.\
                    apply_boundary_remaining(sess, new_grads, net_stru.X, net_stru.Y, length_0, length_1, net_stru.logits, formu, train_set_X,
                                            train_set_Y, to_be_appended_boundary_remaining_points_number)
                # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
                #                         train_set_Y, lower_bound,upper_bound,10+i)
                print("new training size after boundary remaining", "X: ", len(train_set_X))

            for epoch in range(training_epochs):
                _, c = sess.run([net_stru.train_op, net_stru.loss_op], feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})
                # print ("loss: ",c)
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, lower_bound,upper_bound,i)
            train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
            test_y = sess.run(net_stru.logits, feed_dict={net_stru.X: test_set_X})

            train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            test_acc = util.calculate_accuracy(test_y, test_set_Y, False)

            ## save model
            if len(train_acc_list)==0:
                saver.save(sess,save_path)
                train_acc_max=train_acc

            else:
                if train_acc>= train_acc_max:
                    saver.save(sess,save_path)
                    train_acc_max=train_acc
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            threshold=util.calculate_std_dev(train_set_X)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            distance_list, point_pair_list = filter_distant_point_pair(label_0, label_1, threshold)
            util.quickSort(distance_list)
            # print ("sorted list: ",distance_list)
            append_mid_points(distance_list, formu, point_pair_list, to_be_appended_points_number,
                              train_set_X, train_set_Y)

            print("new train size after mid point", len(train_set_X), len(train_set_Y))
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y,lower_bound,upper_bound, 20+i)

            train_set_X, train_set_Y = util.append_random_points(formu, train_set_X, train_set_Y,
                                                                 to_be_appended_random_points_number, lower_bound, upper_bound)
            print ("size after random points:",len(train_set_X))
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, lower_bound,upper_bound,30+i)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)
            print ("weights trained:",sess.run(net_stru.weights))

            # for m in range(initial_point,len(label_0)):
            #     print (label_0[m])
            # print ("weights after :",sess.run(net_stru.weights))


    with tf.Session() as sess:
        saver.restore(sess, save_path)
        # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
        #                             train_set_Y, lower_bound,upper_bound,-1)
        print("weights final:", sess.run(net_stru.weights))
        train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
        test_y = sess.run(net_stru.logits, feed_dict={net_stru.X: test_set_X})

        train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
        test_acc = util.calculate_accuracy(test_y, test_set_Y, False)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    result.append(train_acc_list)
    result.append(test_acc_list)
    print ("Result",result)
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
            num = int(to_be_appended_points_number * 0.4)
            num=int(round(num))
            if(num<1):
                num=1
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)
            pointer = index1
        elif (pointer < index2):
            num = int(to_be_appended_points_number * 0.3)
            num=int(round(num))
            if(num<1):
                num=1
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)
            #
            pointer = index2
        else:
            num = int(to_be_appended_points_number * 0.3)
            num=int(round(num))
            if(num<1):
                num=1
            util.add_distance_values(num, distance_list, selected_distance_list, pointer)
    print ("all list:",distance_list)
    print ("selected list:",selected_distance_list)
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
                        print("from:", point_0, point_1, "middle point:", middle_point, label)
                else:
                    if (middle_point not in train_set_X):
                        train_set_X.append(middle_point)
                        train_set_Y.append([0])
                        print("from:", point_0, point_1, "middle point:", middle_point, label)
