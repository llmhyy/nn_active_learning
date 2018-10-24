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
import communication


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


def generate_accuracy(inputX, inputY, learning_rate, training_epochs,
                      lower_bound, upper_bound, use_bagging, label_tester):
    print("=========MID_POINT===========")
    balance_ratio_threshold = 0.7
    boundary_remaining_trial_iteration = 100

    to_be_appended_points_number = 20
    to_be_appended_boundary_remaining_points_number = 3
    # to_be_appended_random_points_number = 3
    active_learning_iteration = 10
    # threshold = 100
    # training_epochs = 1000

    train_set_X = inputX
    train_set_Y = inputY

    net = ns.NNStructure(train_set_X[0], learning_rate)

    train_acc_list = []

    result = []
    predicted = tf.cast(net.probability > 0.5, dtype=tf.float32)

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))

        # TODO add a to_be_randomed_points_number = 10
        with tf.Session() as sess:
            # sess.run(net.init)
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print(length_0, length_1)
            if (length_0 == 0 or length_1 == 0):
                raise Exception("Cannot be classified")

            # if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
            #     br.apply_boundary_remaining(sess, new_grads, net.X, net.Y, length_0, length_1,
            #                                 net.probability, formu, train_set_X,
            #                                 train_set_Y, to_be_appended_boundary_remaining_points_number, type,
            #                                 name_list, mock)
            #     util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}), train_set_X,
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
                aggregated_network, train_acc = train_bootstrap_model(all_biases_dict, all_data_X, all_data_Y,
                                                                      all_weights_dict, net, parts_num, sess,
                                                                      train_set_X, train_set_Y, training_epochs)
            else:
                util.reset_random_seed()
                sess.run(net.init)
                for epoch in range(training_epochs):
                    _, c = sess.run([net.train_op, net.loss_op],
                                    feed_dict={net.X: train_set_X, net.Y: train_set_Y})
                aggregated_network = net

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
            # append_mid_points(sess, aggregated_network, pair_list, to_be_appended_points_number,
            #                   train_set_X, train_set_Y, label_tester)

            train_set_X, train_set_Y = append_generalization_validation_points(sess, aggregated_network,
                                                                               train_set_X, train_set_Y, 3, label_tester)
            print("new train size after mid point", len(train_set_X), len(train_set_Y))

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

    communication.send_training_finish_message()
    result.append(train_acc_list)
    print("Result", result)
    tf.reset_default_graph()
    return result


def train_bootstrap_model(all_biases_dict, all_data_X, all_data_Y, all_weights_dict, net, parts_num, sess, train_set_X,
                          train_set_Y, training_epochs):
    for parts in range(parts_num):
        best_accuracy = 0
        sess.run(net.init)
        for epoch in range(training_epochs):
            _, c = sess.run([net.train_op, net.loss_op],
                            feed_dict={net.X: all_data_X[parts], net.Y: all_data_Y[parts]})
            train_y = sess.run(net.probability, feed_dict={
                net.X: train_set_X})
            train_acc = util.calculate_accuracy(
                train_y, train_set_Y, False)

        # predicted = tf.cast(net.logits > 0.5, dtype=tf.float32)
        # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
        #                             train_set_X, train_set_Y,
        #                             lower_bound, upper_bound, -1)

        weights_dict = sess.run(net.weights)
        bias_dict = sess.run(net.biases)

        all_weights_dict.append(weights_dict)
        all_biases_dict.append(bias_dict)
    aggregated_network = ns.AggregateNNStructure(train_set_X[0], all_weights_dict, all_biases_dict)
    sess.run(aggregated_network.init)
    train_y = sess.run(aggregated_network.probability, feed_dict={
        aggregated_network.X: train_set_X})
    train_acc = util.calculate_accuracy(train_y, train_set_Y, print_data_details=False)
    print("Bagging performance", "train_acc", train_acc)
    return aggregated_network, train_acc


def append_generalization_validation_points(sess, aggregated_network,
                                            train_set_X, train_set_Y, border_point_number, label_tester):
    # pass in argument n

    # trained model (passed in as argument)
    # cluster according to trainset (threshold)
    # move points away from cluster center in small gradient direction to get new points(if any)
    # re-train the model
    label0 = []
    label1 = []
    for i in range(len(train_set_Y)):
        if train_set_Y[i] == [0]:
            label0.append(train_set_X[i])
        else:
            label1.append(train_set_X[i])

    centers1, border_points_groups1 = cluster.cluster_points(label1, border_point_number)
    centers0, border_points_groups0 = cluster.cluster_points(label0, border_point_number)

    centers = centers0 + centers1
    border_points_groups = border_points_groups0 + border_points_groups1

    print(centers)
    print(border_points_groups)
    gradient = tf.gradients(aggregated_network.probability, aggregated_network.X)

    appended_x = []

    for i in range(len(centers)):
        border_points = border_points_groups[i]
        center = centers[i]

        std_dev = util.calculate_std_dev(border_points)
        step = random.uniform(0, std_dev / 2.0)
        print("step", step)
        for k in range(len(border_points)):
            border_point = border_points[k]
            tmp_vector = []
            for m in range(len(border_point)):
                tmp_value = border_point[m] - center[m]
                tmp_vector.append(tmp_value)

            g = sess.run(gradient, feed_dict={aggregated_network.X: [border_point]})[0]
            decided_gradient = br.decide_all_gradients_for_boundary_remaining(aggregated_network.X, g, [border_point],
                                                                              aggregated_network.probability, sess)

            print("center", center, "point", border_points_groups[i][k])
            print(decided_gradient)
            print(tmp_vector)
            angle = util.calculate_vector_angle(decided_gradient[0], tmp_vector)

            print("angle", angle)

            # move the point
            if angle > -45 and angle < 45:
                gradient_length = util.calculate_vector_size(decided_gradient[0])
                tmp_point = []
                for j in range(len(border_point)):
                    tmp_value = border_point[j] + decided_gradient[0][j] * (step / gradient_length)
                    tmp_point.append(tmp_value)
                print(tmp_point)
                input_point = []
                input_point.append(tmp_point)
                print("input point", input_point)
                result = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: input_point})

                print("result", result)
                if result[0][0] < 0.4:
                    appended_x.append(input_point[0])
                elif result[0][0] > 0.6:
                    appended_x.append(input_point[0])

    labels = label_tester.test_label(appended_x)
    appended_y = []
    for label in labels:
        appended_y.append([label])

    train_set_X = train_set_X + appended_x
    train_set_Y = train_set_Y + appended_y

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


def append_mid_points(sess, aggregated_network, pair_list, to_be_appended_points_number,
                      train_set_X, train_set_Y, label_tester):
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

    results = label_tester.test_label(unconfident_points)

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
