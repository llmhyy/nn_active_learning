from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

import cluster
import communication
import data_pair
import network_structure as ns
import util


def partition_data(label_0, label_1, parts_num):
    result_x = []
    result_y = []
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
        result_x.append(tmpX)
        result_y.append(tmpY)
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

            result_x.append(tmpX)
            result_y.append(tmpY)

    return result_x, result_y


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
            p = data_pair.DataPair(m, n)
            if p.distance > threshold:
                pair = p
                pair_list.append(pair)

    return pair_list


def generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate, training_epochs,
                      lower_bound, upper_bound, use_bagging, label_tester, point_number_limit, model_folder,
                      model_file):
    print("=========MID_POINT===========")

    mid_point_limit = 10
    generalization_valid_limit = 10

    net = ns.NNStructure(len(train_set_x[0]), learning_rate)
    aggregated_network = None

    train_acc_list = []
    test_acc_list = []
    data_point_number_list = []
    appended_point_list = []

    predicted = tf.cast(net.probability > 0.5, dtype=tf.float32)

    count = 1
    total_appended_x = []
    total_appended_y = []
    while len(train_set_x) < point_number_limit:
        print("*******", count, "th loop:")
        print("training set size", len(train_set_x))

        with tf.Session() as sess:
            label_0, label_1 = util.data_partition(train_set_x, train_set_y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print(length_0, length_1)
            if length_0 == 0 or length_1 == 0:
                raise Exception("Cannot be classified")

            smaller_set_size = min(len(label_0), len(label_1))
            larger_set_size = max(len(label_0), len(label_1))
            parts_num = int(larger_set_size / smaller_set_size)

            all_data_x, all_data_y = partition_data(label_0, label_1, parts_num)
            tmp = list(zip(all_data_x, all_data_y))
            random.shuffle(tmp)
            all_data_x, all_data_y = zip(*tmp)

            if use_bagging:
                aggregated_network, train_acc = train_bootstrap_model(all_data_x, all_data_y,
                                                                      total_appended_x,
                                                                      total_appended_y,
                                                                      net, parts_num, sess,
                                                                      train_set_x, train_set_y, training_epochs,
                                                                      lower_bound, upper_bound)
            else:
                sess.run(net.w_init)
                for epoch in range(training_epochs):
                    _, c = sess.run([net.train_op, net.loss_op],
                                    feed_dict={net.X: train_set_x, net.Y: train_set_y})

                util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                                            train_set_x, train_set_y, lower_bound, upper_bound, count)

                # if len(total_appended_x) > 0:
                #     for epoch in range(5):
                #         _, c = sess.run([net.train_op, net.loss_op],
                #                         feed_dict={net.X: total_appended_x, net.Y: total_appended_y})
                #     util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                #                                 train_set_x, train_set_y, lower_bound, upper_bound, count)
                aggregated_network = net

            train_y = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: train_set_x})
            train_acc = util.calculate_accuracy(train_y, train_set_y, False)

            test_y = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: test_set_x})
            test_acc = util.calculate_accuracy(test_y, test_set_y, False)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            data_point_number_list.append(len(train_set_x))

            # new_grads = tf.gradients(net.probability, net.X)
            # appended_x, appended_y = br.apply_boundary_remaining(sess, new_grads, net.X, net.Y, length_0, length_1,
            #                                                      net.probability, train_set_x,
            #                                                      train_set_y,
            #                                                      to_be_appended_boundary_remaining_points_number,
            #                                                      label_tester)

            predicted = tf.cast(aggregated_network.probability > 0.5, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={aggregated_network.X: x}),
                                        train_set_x, train_set_y, lower_bound, upper_bound, count)

            if len(train_set_x) > point_number_limit:
                break

            total_appended_x = []
            total_appended_y = []

            std_dev = util.calculate_std_dev(train_set_x)
            centers, centers_label, clusters, border_points_groups = cluster_training_data(train_set_x, train_set_y, 2,
                                                                                           5)

            appending_dict = {}
            appended_x, appended_y = append_generalization_validation_points(sess, aggregated_network, lower_bound,
                                                                             upper_bound, std_dev,
                                                                             label_tester, centers, centers_label,
                                                                             clusters, border_points_groups,
                                                                             generalization_valid_limit, train_set_x)
            total_appended_x += appended_x
            total_appended_y += appended_y
            appending_dict["generalization_validation"] = appended_x

            pair_list = select_point_pair(centers, centers_label, clusters, mid_point_limit)
            appended_x, appended_y = append_mid_points(sess, aggregated_network, pair_list,
                                                       train_set_x, train_set_y, label_tester)
            total_appended_x += appended_x
            total_appended_y += appended_y
            train_set_x += total_appended_x
            train_set_y += total_appended_y

            appending_dict["mid_point"] = appended_x
            appended_point_list.append(appending_dict)

            label_0, label_1 = util.data_partition(train_set_x, train_set_y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

            util.save_model(sess, model_folder, model_file)
            count += 1

    communication.send_training_finish_message()
    # tf.reset_default_graph()
    return train_acc_list, test_acc_list, data_point_number_list, appended_point_list


def select_point_pair(centers, centers_label, clusters, mid_point_limit):
    positive_clusters = []
    negative_clusters = []
    for i in range(len(centers_label)):
        if centers_label[i]:
            positive_clusters.append(clusters[i])
        else:
            negative_clusters.append(clusters[i])
    max_per_pair_num = (int)(mid_point_limit / len(positive_clusters))

    final_pair_list = []
    for positive_cluster in positive_clusters:
        positive_point = random.choice(positive_cluster)

        pair_list = []
        for negative_cluster in negative_clusters:
            negative_point = random.choice(negative_cluster)
            p = data_pair.DataPair(negative_point, positive_point)
            pair_list.append(p)

        pair_list.sort(key=lambda x: x.distance)
        if len(pair_list) <= max_per_pair_num:
            final_pair_list += pair_list
        else:
            final_pair_list += pair_list[0:2]

    return final_pair_list


def train_bootstrap_model(all_data_x, all_data_y, total_appended_x, total_appended_y, net, parts_num, sess, train_set_x,
                          train_set_y, training_epochs, lower_bound, upper_bound):
    all_biases_dict = []
    all_weights_dict = []
    for iteration in range(parts_num):
        sess.run(net.init)
        for epoch in range(training_epochs):
            _, c = sess.run([net.train_op, net.loss_op],
                            feed_dict={net.X: all_data_x[iteration], net.Y: all_data_y[iteration]})
            sess.run(net.probability, feed_dict={net.X: train_set_x})

        predicted = tf.cast(net.logits > 0.5, dtype=tf.float32)
        util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                                    all_data_x[iteration], all_data_y[iteration],
                                    lower_bound, upper_bound, 100 + iteration)

        weights_dict = sess.run(net.weights)
        bias_dict = sess.run(net.biases)

        all_weights_dict.append(weights_dict)
        all_biases_dict.append(bias_dict)

    aggregated_network = ns.AggregateNNStructure(len(train_set_x[0]), all_weights_dict, all_biases_dict)

    sess.run(aggregated_network.init)
    train_y = sess.run(aggregated_network.probability, feed_dict={
        aggregated_network.X: train_set_x})
    train_acc = util.calculate_accuracy(train_y, train_set_y, print_data_details=False)

    return aggregated_network, train_acc


def cluster_training_data(train_set_x, train_set_y, border_point_number, cluster_number):
    label0 = []
    label1 = []
    for i in range(len(train_set_y)):
        if train_set_y[i] == [0]:
            label0.append(train_set_x[i])
        else:
            label1.append(train_set_x[i])

    centers1, border_points_groups1, clusters1 = cluster.cluster_points(label1, border_point_number, cluster_number)
    util.plot_clustering_result(clusters1, -1000, 1000, 1)
    centers_label1 = np.ones(len(centers1)).tolist()

    centers0, border_points_groups0, clusters0 = cluster.cluster_points(label0, border_point_number, cluster_number)
    util.plot_clustering_result(clusters0, -1000, 1000, 2)
    centers_label0 = np.zeros(len(centers0)).tolist()

    centers = centers0 + centers1
    centers_label = centers_label0 + centers_label1
    clusters = clusters0 + clusters1
    border_points_groups = border_points_groups0 + border_points_groups1

    # centers = centers1
    # centers_label = centers_label1
    # border_points_groups = border_points_groups1

    return centers, centers_label, clusters, border_points_groups


def append_generalization_validation_points(sess, aggregated_network, lower_bound, upper_bound, std_dev,
                                            label_tester, centers, centers_label, clusters, border_points_groups,
                                            generalization_valid_limit, train_set_x):
    # pass in argument n

    gradient = tf.gradients(aggregated_network.probability, aggregated_network.X)

    appended_x = search_validation_points(aggregated_network, border_points_groups, centers, centers_label, clusters,
                                          gradient, sess, lower_bound, upper_bound, std_dev, generalization_valid_limit,
                                          train_set_x)

    appended_y = []
    if len(appended_x) != 0:
        labels = label_tester.test_label(appended_x)
        for label in labels:
            appended_y.append([label])

    return appended_x, appended_y


def search_validation_points(aggregated_network, border_points_groups, centers, centers_label, clusters,
                             gradient, sess, lower_bound, upper_bound, std_dev, generalization_valid_limit,
                             train_set_x):
    appended_x = []
    for i in range(len(centers)):
        border_points = border_points_groups[i]
        center = centers[i]
        label = centers_label[i]
        single_cluster = clusters[i]

        cluster_dev = util.calculate_std_dev(single_cluster)
        if cluster_dev == 0:
            step = random.uniform(0, std_dev)
        else:
            step = random.uniform(0, cluster_dev)

        for k in range(len(border_points)):
            if len(appended_x) > generalization_valid_limit:
                break

            border_point = border_points[k]
            original_border_point = border_point
            decided_direction = calculate_decided_direction(aggregated_network, border_point,
                                                            center, gradient, sess)
            # move the point
            best_point = []
            move_count = 0
            while move_count < 10:
                gradient_length = util.calculate_vector_size(decided_direction[0])
                new_point = []
                for j in range(len(border_point)):
                    new_value = border_point[j] + decided_direction[0][j] * (step / gradient_length)
                    new_point.append(new_value)

                probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [new_point]})

                if is_point_valid(new_point, probability, lower_bound, upper_bound, label):
                    best_point = new_point
                else:
                    break

                decided_direction = calculate_decided_direction(aggregated_network, new_point, center,
                                                                gradient, sess)

                border_point = new_point
                move_count = move_count + 1

            if len(best_point) > 0:
                if not is_too_close(train_set_x, best_point):
                    appended_x.append(best_point)

                # mid_point = ((np.array(best_point) + np.array(original_border_point))/2).tolist()
                # appended_x.append(mid_point)
    return appended_x


def is_too_close(single_cluster, best_point):
    return False
    # for point in single_cluster:
    #     distance = util.calculate_distance(point, best_point)
    #     if distance < 10:
    #         return True
    #
    # return False


def confirm_gradient_direction(sess, point, aggregated_network, gradient):
    delta = 10E-6
    point_add = []
    point_minus = []

    for i in range(len(point)):
        value = point[i]
        g = gradient[0][i]
        value_add = value + delta * g
        value_minus = value - delta * g

        point_add.append(value_add)
        point_minus.append(value_minus)

    y = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [point]})[0]
    y_add = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [point_add]})[0]
    y_minus = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [point_minus]})[0]

    if y < 0.5:
        if y_add > y_minus:
            return gradient
        else:
            return -gradient
    else:
        if y_add > y_minus:
            return -gradient
        else:
            return gradient


def calculate_decided_direction(aggregated_network, point, center, gradient, sess):
    vector = util.calculate_direction(point, center)
    vector_length = util.calculate_vector_size(vector)
    g = sess.run(gradient, feed_dict={aggregated_network.X: [point]})[0]
    g = confirm_gradient_direction(sess, point, aggregated_network, g)
    g_length = util.calculate_vector_size(g[0].tolist())

    if vector_length == 0 and g_length == 0:
        direction = np.random.randn(len(point)).tolist()
    elif vector_length == 0 and g_length != 0:
        direction = util.calculate_orthogonal_direction(g[0].tolist())
    elif vector_length != 0 and g_length == 0:
        direction = vector
        print("point", point, "leaving center direction", direction, "center is", center)
        pass
    else:
        direction = util.calculate_vector_projection(vector, g[0].tolist())
        print("point", point, "move towards direction", direction, "center is", center)
        pass

    decided_gradient = [direction]

    return decided_gradient


def is_point_valid(new_point, probability, lower_bound, upper_bound, label):
    if probability[0][0] < 0.4 or probability[0][0] > 0.6:
        if probability[0][0] < 0.4 and label == 1:
            return False
        elif probability[0][0] > 0.6 and label == 0:
            return False

        for value in new_point:
            if lower_bound > value or value > upper_bound:
                return False
        return True

    return False


def calculate_unconfident_mid_point(sess, aggregated_network, pair):
    px = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [pair.point_x]})[0]
    py = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [pair.point_y]})[0]
    if px > 0.5:
        return None
    elif py <= 0.5:
        return None

    mid_point = pair.calculate_mid_point()
    probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [mid_point]})

    while probability < 0.4 or probability > 0.6:
        if probability < 0.5:
            pair = data_pair.DataPair(mid_point, pair.point_y)
        else:
            pair = data_pair.DataPair(pair.point_x, mid_point)
        mid_point = pair.calculate_mid_point()
        probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [mid_point]})

    return mid_point


def append_mid_points(sess, aggregated_network, pair_list,
                      train_set_x, train_set_y, label_tester):
    unconfident_points = []
    for pair in pair_list:
        point = calculate_unconfident_mid_point(sess, aggregated_network, pair)
        if point is not None:
            if not (point in unconfident_points):
                if not is_too_close(train_set_x, point):
                    unconfident_points.append(point)
            else:
                # print()
                pass

    print("sampled mid points", unconfident_points)

    appended_x = []
    appended_y = []
    if len(unconfident_points) != 0:
        results = label_tester.test_label(unconfident_points)
        for i in range(len(results)):
            result = results[i]
            middle_point = unconfident_points[i]
            if middle_point not in train_set_x:
                appended_x.append(middle_point)
                appended_y.append([result])
            # probability = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: [middle_point]})[0]
            # if result == 1 and probability < 0.5:
            #     if middle_point not in train_set_x:
            #         appended_x.append(middle_point)
            #         appended_y.append([result])
            # elif result == 0 and probability > 0.5:
            #     if middle_point not in train_set_x:
            #         appended_x.append(middle_point)
            #         appended_y.append([result])

    return appended_x, appended_y
