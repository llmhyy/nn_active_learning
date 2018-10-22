from __future__ import print_function

import operator
import random

import tensorflow as tf

import boundary_remaining as br
import cluster
import data_pair
import network_structure as ns
import testing_function
import util


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
            p = data_pair.DataPair(m, n, False, True)
            if (p.distance > threshold):
                pair = p
                pair_list.append(pair)

    return pair_list


def generate_accuracy(inputX, inputY, train_data_file, test_data_file, formula, learning_rate, training_epochs,
                      lower_bound, upper_bound, use_bagging, data_type, name_list, mock):
    print("=========MID_POINT===========")
    balance_ratio_threshold = 0.7
    boundary_remaining_trial_iteration = 100

    to_be_appended_points_number = 20
    to_be_appended_boundary_remaining_points_number = 3
    # to_be_appended_random_points_number = 3
    active_learning_iteration = 10
    # threshold = 100
    # training_epochs = 1000

    train_set_X = []
    train_set_Y = []
    test_set_X = []
    test_set_Y = []
    if mock:
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

    new_grads = tf.gradients(net_stru.probability, net_stru.X)
    y = None

    predicted = tf.cast(net_stru.probability > 0.5, dtype=tf.float32)

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))

        # TODO add a to_be_randomed_points_number = 10

        with tf.Session() as sess:
            # sess.run(net_stru.init)
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

            # if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
            #     br.apply_boundary_remaining(sess, new_grads, net_stru.X, net_stru.Y, length_0, length_1,
            #                                 net_stru.probability, formu, train_set_X,
            #                                 train_set_Y, to_be_appended_boundary_remaining_points_number, type,
            #                                 name_list, mock)
            #     util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                                 train_set_Y, lower_bound, upper_bound, 10 + i)
            #     print("new training size after boundary remaining", "X: ", len(train_set_X))

            # print(train_set_X)
            # print(train_set_Y)
            smaller_set_size = min(len(label_0), len(label_1))
            larger_set_size = max(len(label_0), len(label_1))
            parts_num = int(larger_set_size / smaller_set_size)
            # parts_num = 2

            all_data_X, all_data_Y = partition_data(label_0, label_1, parts_num)
            tmp = list(zip(all_data_X, all_data_Y))
            random.shuffle(tmp)
            all_data_X, all_data_Y = zip(*tmp)

            all_weights_dict = []
            all_biases_dict = []

            if use_bagging:
                for parts in range(parts_num):
                    best_accuracy = 0
                    sess.run(net_stru.init)
                    for epoch in range(training_epochs):
                        _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                        feed_dict={net_stru.X: all_data_X[parts], net_stru.Y: all_data_Y[parts]})
                        train_y = sess.run(net_stru.probability, feed_dict={
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

                aggregated_network = ns.AggregateNNStructure(train_set_X[0], all_weights_dict, all_biases_dict)
                sess.run(aggregated_network.init)

                train_y = sess.run(aggregated_network.probability, feed_dict={
                    aggregated_network.X: train_set_X})
                train_acc = util.calculate_accuracy(train_y, train_set_Y, print_data_details=False)
                print("Bagging performance", "train_acc", train_acc)

            else:
                util.reset_random_seed()
                sess.run(net_stru.init)
                for epoch in range(training_epochs):
                    _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                    feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})
                aggregated_network = net_stru

                train_y = sess.run(aggregated_network.probability, feed_dict={
                    aggregated_network.X: train_set_X})
                train_acc = util.calculate_accuracy(
                    train_y, train_set_Y, print_data_details=False)
                print("train_acc", train_acc)

            train_acc_list.append(train_acc)
            threshold = util.calculate_std_dev(train_set_X)

            predicted = tf.cast(aggregated_network.probability > 0.5, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={aggregated_network.X: x}),
                                        train_set_X, train_set_Y, lower_bound, upper_bound, i)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            pair_list = filter_distant_point_pair(label_0, label_1, threshold)
            sorted(pair_list, key=operator.attrgetter('distance'))
            # append_mid_points(sess, aggregated_network, pair_list, formu, to_be_appended_points_number,
            #                   train_set_X, train_set_Y, type, name_list, mock)

            # append_extrapolated_points(sess, aggregated_network)
            train_set_X, train_set_Y = append_generalization_validation_points(sess, aggregated_network, formula,
                                                                               train_set_X, train_set_Y, data_type,
                                                                               name_list, mock, 10)
            print("new train size after mid point", len(train_set_X), len(train_set_Y))

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

    print("$TRAINING_FINISH")

    result.append(train_acc_list)
    result.append(test_acc_list)
    print("Result", result)
    tf.reset_default_graph()
    return result


def append_generalization_validation_points(sess, aggregated_network, formu,
                                            train_set_X, train_set_Y, type, name_list, mock, n):
    # pass in argument n

    # trained model (passed in as argument)
    # cluster according to trainset (threshold)
    # move points away from cluster center in small gradient direction to get new points(if any)
    # re-train the model
    n = 10
    label0 = []
    label1 = []
    for i in range(len(train_set_Y)):
        if train_set_Y[i] == [0]:
            label0.append(train_set_X[i])
        else:
            label1.append(train_set_X[i])

    centers1, n_largest1 = cluster.cluster_points(label1, n)
    centers0, n_largest0 = cluster.cluster_points(label0, n)

    centers = centers0 + centers1
    n_largest = n_largest0 + n_largest1

    print(centers)
    print(n_largest)
    gradient = tf.gradients(aggregated_network.probability, aggregated_network.X)

    outputX = []
    outputY = []

    for i in range(len(centers)):
        std_dev = util.calculate_std_dev(n_largest[i])
        step = random.uniform(0, std_dev / 2.0)
        print("step", step)
        for k in range(len(n_largest[i])):

            tmp_vector = []
            for m in range(len(n_largest[i][k])):
                tmp_value = n_largest[i][k][m] - centers[i][m]
                tmp_vector.append(tmp_value)

            input = []
            input.append(n_largest[i][k])
            g = sess.run(gradient, feed_dict={aggregated_network.X: input})[0]
            decided_gradient = br.decide_all_gradients_for_boundary_remaining(aggregated_network.X, g, input,
                                                                              aggregated_network.probability, sess)

            print("center", centers[i], "point", n_largest[i][k])
            print(decided_gradient)
            print(tmp_vector)
            angle = util.calculate_vector_angle(decided_gradient[0], tmp_vector)

            print("angle", angle)

            # move the point
            if angle > -45 and angle < 45:
                gradient_length = util.calculate_vector_size(decided_gradient[0])
                tmp_point = []
                for j in range(len(n_largest[i][k])):
                    tmp_value = n_largest[i][k][j] + decided_gradient[0][j] * (step / gradient_length)
                    tmp_point.append(tmp_value)
                print(tmp_point)
                input_point = []
                input_point.append(tmp_point)
                print("input point", input_point)
                result = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: input_point})

                print("result", result)
                if result[0][0] < 0.4:
                    outputX.append(input_point[0])
                    outputY.append([0])

                elif result[0][0] > 0.6:
                    outputX.append(input_point[0])
                    outputY.append([1])
    print(outputX)
    print(outputY)

    train_set_X = train_set_X + outputX
    train_set_Y = train_set_Y + outputY

    return train_set_X, train_set_Y


# def generate_random_points(train_set_X,train_set_Y,to_be_appended_critical_points_number):
#     length=len(train_set_X)-1
#     random_index_list=[]
#     outputX=[]
#     outputY=[]
#     count=0
#     while count<to_be_appended_critical_points_number:
#         random_number = random.randint(length)
#         if random_number in random_index_list:
#             continue
#         else:
#             random_index_list.append(random_number)
#         pointX=train_set_X[random_number]
#         std_dev=util.calculate_std_dev(train_set_X)
#         smallest_distance=calculate_smallest_distance(train_set_X,pointX)
#
#         if smallest_distance<std_dev:
#             continue
#
#         outputX.append(pointX)
#         outputY.append(train_set_Y[random_number])
#         count+=1
#
#
#     return outputX,outputY

# def calculate_smallest_distance(train_set_X,pointX):
#     smallest_distance=math.inf
#     dimension = len(train_set_X[0])
#     for i in range(len(train_set_X)):
#         if train_set_X[i][0]!=pointX[0]:
#             distance=0
#             for d in dimension:
#                 distance+=(train_set_X[i][d]-pointX[d])**2
#             distance = math.sqrt(distance)
#             if distance<smallest_distance:
#                 smallest_distance=distance
#     return distance

def calculate_unconfident_mid_point(sess, aggregated_network, pair):
    px = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [pair.point_x]})[0]
    py = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [pair.point_y]})[0]
    predict_x = px > 0.5
    predict_y = py > 0.5
    if predict_x or not predict_y:
        if predict_x:
            return pair.point_x
        elif not predict_y:
            return pair.point_y

    mid_point = pair.calculate_mid_point()
    list = []
    list.append(mid_point)
    probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: list})

    while probability < 0.4 or probability > 0.6:
        if probability < 0.5:
            pair = data_pair.DataPair(mid_point, pair.point_y, False, True)
        else:
            pair = data_pair.DataPair(pair.point_x, mid_point, False, True)
        mid_point = pair.calculate_mid_point()
        list = []
        list.append(mid_point)
        probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: list})

    return mid_point


def append_mid_points(sess, aggregated_network, pair_list, formu, to_be_appended_points_number,
                      train_set_X, train_set_Y, type, name_list, mock):
    selected_pairs = []
    # to_be_appended_points_number = len(pair_list)
    if to_be_appended_points_number > len(pair_list):
        to_be_appended_points_number = len(pair_list)
    for i in range(to_be_appended_points_number):
        index = random.randint(0, len(pair_list) - 1)
        pair = pair_list[index]
        selected_pairs.append(pair)
        pair_list.remove(pair)

    unconfident_points = []
    for pair in selected_pairs:
        point = calculate_unconfident_mid_point(sess, aggregated_network, pair)
        if (point is not None):
            if not (point in unconfident_points):
                unconfident_points.append(point)
            else:
                # print()
                pass

    print("sampled mid points", unconfident_points)

    results = testing_function.test_label(unconfident_points, formu, type, name_list, mock)

    for i in range(len(results)):
        result = results[i]
        middle_point = unconfident_points[i]
        probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [middle_point]})[0]
        if result == 1 and probability < 0.5:
            if middle_point not in train_set_X:
                train_set_X.append(middle_point)
                train_set_Y.append([1])
        elif result == 0 and probability > 0.5:
            if (middle_point not in train_set_X):
                train_set_X.append(middle_point)
                train_set_Y.append([0])

    # lenlength = len(pair_list)
    # index1 = int(length / 3)
    # index2 = int(length / 3 * 2)
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
