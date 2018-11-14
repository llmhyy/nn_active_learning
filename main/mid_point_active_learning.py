from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from main import data_pair, communication, util, cluster, network_structure as ns, gradient_util


class MidPointActiveLearner:
    def __init__(self, train_set_x_info, train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                 training_epochs,
                 lower_bound, upper_bound, use_bagging, label_tester, point_number_limit,
                 model_folder,
                 model_file, mid_point_limit = 10, generalization_valid_limit = 10):
        self.train_set_x_info = train_set_x_info
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        self.test_set_x = test_set_x
        self.test_set_y = test_set_y
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.use_bagging = use_bagging
        self.label_tester = label_tester
        self.point_number_limit = point_number_limit
        self.model_folder = model_folder
        self.model_file = model_file
        self.mid_point_limit = mid_point_limit
        self.generalization_valid_limit = generalization_valid_limit

    def partition_data(self, label_0, label_1, parts_num):
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

    def filter_distant_point_pair(self, label_0, label_1, threshold):
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

    def train(self):
        print("=========MID_POINT===========")
        # tf.reset_default_graph()
        train_acc_list = []
        test_acc_list = []
        data_point_number_list = []
        appended_point_list = []

        count = 1
        while len(self.train_set_x) < self.point_number_limit:
            tf.reset_default_graph()
            util.reset_random_seed()
            net = ns.NNStructure(len(self.train_set_x[0]), self.learning_rate)

            aggregated_network = None
            predicted = tf.cast(net.probability > 0.5, dtype=tf.float32)

            print("*******", count, "th loop:")
            print("training set size", len(self.train_set_x))

            with tf.Session() as sess:
                label_0, label_1 = util.data_partition(self.train_set_x, self.train_set_y)

                # path = "./dataset/data" + str(count) + ".csv"
                # dg.write_to_file(train_set_x, train_set_y, path)

                length_0 = len(label_0) + 0.0
                length_1 = len(label_1) + 0.0

                print(length_0, length_1)
                if length_0 == 0 or length_1 == 0:
                    raise Exception("Cannot be classified")

                if self.use_bagging:
                    smaller_set_size = min(len(label_0), len(label_1))
                    larger_set_size = max(len(label_0), len(label_1))
                    parts_num = int(larger_set_size / smaller_set_size)
                    all_data_x, all_data_y = self.partition_data(label_0, label_1, parts_num)
                    tmp = list(zip(all_data_x, all_data_y))
                    random.shuffle(tmp)
                    all_data_x, all_data_y = zip(*tmp)
                    aggregated_network, train_acc = self.train_bootstrap_model(all_data_x, all_data_y,
                                                                               net, parts_num, sess)
                else:
                    sess.run(net.init)
                    for epoch in range(self.training_epochs):
                        sess.run([net.train_op, net.loss_op],
                                 feed_dict={net.X: self.train_set_x, net.Y: self.train_set_y})
                    aggregated_network = net

                train_y = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: self.train_set_x})
                train_acc = util.calculate_accuracy(train_y, self.train_set_y, False)
                print("train_acc", train_acc)
                train_acc_list.append(train_acc)

                if self.test_set_x is not None:
                    test_y = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: self.test_set_x})
                    test_acc = util.calculate_accuracy(test_y, self.test_set_y, False)
                    test_acc_list.append(test_acc)
                    print("test_acc", test_acc)

                data_point_number_list.append(len(self.train_set_x))

                predicted = tf.cast(aggregated_network.probability > 0.5, dtype=tf.float32)
                util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={aggregated_network.X: x}),
                                            self.train_set_x, self.train_set_y, self.lower_bound, self.upper_bound,
                                            count)

                if len(self.train_set_x) > self.point_number_limit:
                    break

                total_appended_x = []
                total_appended_y = []

                std_dev = util.calculate_std_dev(self.train_set_x)

                cluster_number = 3
                # border_number = (int)(generalization_valid_limit/(2*cluster_number))
                centers, centers_label, clusters, border_points_groups = self.cluster_training_data(4, 3)

                appending_dict = {}
                print("start generalization validation")
                appended_x, appended_y = self.append_generalization_validation_points(sess, aggregated_network,
                                                                                      std_dev,
                                                                                      centers,
                                                                                      centers_label,
                                                                                      clusters,
                                                                                      border_points_groups,
                                                                                      )
                total_appended_x += appended_x
                total_appended_y += appended_y
                appending_dict["generalization_validation"] = appended_x

                print("start midpoint selection")
                pair_list = self.select_point_pair(centers, centers_label, clusters)
                appended_x, appended_y = self.append_mid_points(sess, aggregated_network, pair_list)
                total_appended_x += appended_x
                total_appended_y += appended_y

                self.train_set_x += total_appended_x
                self.train_set_y += total_appended_y

                appending_dict["mid_point"] = appended_x
                appended_point_list.append(appending_dict)

                label_0, label_1 = util.data_partition(self.train_set_x, self.train_set_y)
                length_0 = len(label_0) + 0.0
                length_1 = len(label_1) + 0.0

                print("label 0 length", length_0, "label 1 length", length_1)

                util.save_model(sess, self.model_folder, self.model_file)
                count += 1

        communication.send_training_finish_message()
        return train_acc_list, test_acc_list, data_point_number_list, appended_point_list

    def select_point_pair(self, centers, centers_label, clusters):
        positive_clusters = []
        negative_clusters = []
        for i in range(len(centers_label)):
            if centers_label[i]:
                positive_clusters.append(clusters[i])
            else:
                negative_clusters.append(clusters[i])
        max_per_pair_num = (int)(self.mid_point_limit / len(positive_clusters))

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

        # if len(final_pair_list) < mid_point_limit:
        #     iterations = mid_point_limit - len(final_pair_list)
        #     for i in range(iterations):
        #         positive_cluster = random.choice(positive_clusters)
        #         positive_point = random.choice(positive_cluster)
        #         negative_cluster = random.choice(negative_clusters)
        #         negative_point = random.choice(negative_cluster)
        #
        #         p = data_pair.DataPair(negative_point, positive_point)
        #         final_pair_list.append(p)

        return final_pair_list

    def train_bootstrap_model(self, all_data_x, all_data_y, net, parts_num, sess):
        all_biases_dict = []
        all_weights_dict = []
        for iteration in range(parts_num):
            sess.run(net.init)
            for epoch in range(self.training_epochs):
                _, c = sess.run([net.train_op, net.loss_op],
                                feed_dict={net.X: all_data_x[iteration], net.Y: all_data_y[iteration]})
                sess.run(net.probability, feed_dict={net.X: self.train_set_x})

            predicted = tf.cast(net.logits > 0.5, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                                        all_data_x[iteration], all_data_y[iteration],
                                        self.lower_bound, self.upper_bound, 100 + iteration)

            weights_dict = sess.run(net.weights)
            bias_dict = sess.run(net.biases)

            all_weights_dict.append(weights_dict)
            all_biases_dict.append(bias_dict)

        aggregated_network = ns.AggregateNNStructure(len(self.train_set_x[0]), all_weights_dict, all_biases_dict)

        sess.run(aggregated_network.init)
        train_y = sess.run(aggregated_network.probability, feed_dict={
            aggregated_network.X: self.train_set_x})
        train_acc = util.calculate_accuracy(train_y, self.train_set_y, print_data_details=False)

        return aggregated_network, train_acc

    def cluster_training_data(self, border_point_number, cluster_number):
        label0 = []
        label1 = []
        for i in range(len(self.train_set_y)):
            if self.train_set_y[i] == [0]:
                label0.append(self.train_set_x[i])
            else:
                label1.append(self.train_set_x[i])

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

    def append_generalization_validation_points(self, sess, aggregated_network, std_dev,
                                                centers, centers_label, clusters,
                                                border_points_groups):
        # pass in argument n

        gradient = tf.gradients(aggregated_network.probability, aggregated_network.X)

        appended_x = self.search_validation_points(aggregated_network, border_points_groups, centers,
                                                   centers_label, clusters,
                                                   gradient, sess, std_dev)
        appended_x = util.convert_with_data_type_and_mask(appended_x, self.train_set_x_info, self.label_tester)

        appended_y = []
        if len(appended_x) != 0:
            labels = self.label_tester.test_label(appended_x)
            for label in labels:
                appended_y.append([label])

        return appended_x, appended_y

    def random_step(self, cluster_dev, std_dev):
        if cluster_dev == 0:
            step = random.uniform(0, std_dev)
        else:
            step = random.uniform(0, cluster_dev)
        return step

    def search_validation_points(self, aggregated_network, border_points_groups, centers, centers_label, clusters,
                                 gradient, sess, std_dev):
        zip_list = list(zip(border_points_groups, centers, centers_label, clusters))
        random.shuffle(zip_list)
        border_points_groups, centers, centers_label, clusters = zip(*zip_list)

        appended_x = []
        for i in range(len(centers)):
            border_points = border_points_groups[i]
            center = centers[i]
            label = centers_label[i]
            single_cluster = clusters[i]
            cluster_dev = util.calculate_std_dev(single_cluster)
            step = self.random_step(cluster_dev, std_dev)
            for k in range(len(border_points)):
                if len(appended_x) > self.generalization_valid_limit:
                    break

                border_point = border_points[k]
                original_border_point = border_point
                decided_direction = self.calculate_decided_direction(aggregated_network, border_point,
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
                    # new_point = util.convert_with_data_type_and_mask(new_point, self.train_set_x_info, self.label_tester)
                    probability = sess.run(aggregated_network.probability,
                                           feed_dict={aggregated_network.X: [new_point]})

                    if self.is_point_valid(new_point, probability, label):
                        best_point = new_point
                    else:
                        break

                    decided_direction = self.calculate_decided_direction(aggregated_network, new_point, center,
                                                                         gradient, sess)

                    border_point = new_point
                    move_count = move_count + 1

                if len(best_point) > 0:
                    if not self.is_too_close(self.train_set_x, best_point):
                        appended_x.append(best_point)

                    # mid_point = ((np.array(best_point) + np.array(original_border_point))/2).tolist()
                    # appended_x.append(mid_point)
        return appended_x

    def is_too_close(self, single_cluster, best_point):
        return False
        # for point in single_cluster:
        #     distance = util.calculate_distance(point, best_point)
        #     if distance < 10:
        #         return True
        #
        # return False

    def calculate_decided_direction(self, aggregated_network, point, center, gradient, sess):
        vector = util.calculate_direction(point, center)
        vector_length = util.calculate_vector_size(vector)
        g = sess.run(gradient, feed_dict={aggregated_network.X: [point]})
        g = gradient_util.confirm_gradient_direction(sess, [point], aggregated_network, g)[0]
        g_length = util.calculate_vector_size(g[0].tolist())

        if vector_length == 0 and g_length == 0:
            direction = np.random.randn(len(point)).tolist()
        elif vector_length == 0 and g_length != 0:
            direction = util.calculate_orthogonal_direction(g.tolist())
        elif vector_length != 0 and g_length == 0:
            direction = vector
            # print("point", point, "leaving center direction", direction, "center is", center)
            pass
        else:
            direction = util.calculate_vector_projection(vector, g.tolist())
            # print("point", point, "move towards direction", direction, "center is", center)
            pass

        decided_gradient = [direction]

        return decided_gradient

    def is_point_valid(self, new_point, probability, label):
        if probability[0][0] < 0.4 or probability[0][0] > 0.6:
            if probability[0][0] < 0.4 and label == 1:
                return False
            elif probability[0][0] > 0.6 and label == 0:
                return False

            for value in new_point:
                if self.lower_bound > value or value > self.upper_bound:
                    return False
            return True

        return True

    def calculate_unconfident_mid_point(self, sess, aggregated_network, pair):
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

    def append_mid_points(self, sess, aggregated_network, pair_list):
        unconfident_points = []
        for pair in pair_list:
            point = self.calculate_unconfident_mid_point(sess, aggregated_network, pair)
            if point is not None:
                if not (point in unconfident_points):
                    if not self.is_too_close(self.train_set_x, point):
                        unconfident_points.append(point)

        appended_x = []
        appended_y = []
        if len(unconfident_points) != 0:
            unconfident_points = util.convert_with_data_type_and_mask(unconfident_points, self.train_set_x_info,
                                                                      self.label_tester)
            labels = self.label_tester.test_label(unconfident_points)
            for i in range(len(labels)):
                result = labels[i]
                middle_point = unconfident_points[i]
                if middle_point not in self.train_set_x:
                    appended_x.append(middle_point)
                    appended_y.append([result])

        return appended_x, appended_y
