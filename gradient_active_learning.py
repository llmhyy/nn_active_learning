from __future__ import print_function

import tensorflow as tf
import csv

import numpy as np
import random
import boundary_remaining as br
import gradient_combination
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


def calculate_average(weights, biases, weight_tags, bias_tags):
    weights_dict = {}
    biases_dict = {}

    for tag in weight_tags:
        tag = str(tag)
        ave_value = []
        for h in range(len(weights[0][tag])):
            ave_h1_single = []
            for k in range(len(weights[0][tag][0])):
                tmp_value = 0
                for weight in weights:
                    tmp_value += weight[tag][h][k]
                    # value = weight[tag][h][k]
                    # if value < tmp_value:
                    #     tmp_value = weight[tag][h][k]
                tmp_value = tmp_value / len(weights)
                ave_h1_single.append(tmp_value)
            ave_value.append(ave_h1_single)
        # print(tag, ave_value)
        weights_dict[tag] = tf.Variable(ave_value)

    for tag in bias_tags:
        tag = str(tag)
        ave_h1_single = []
        for h in range(len(biases[0][tag])):
            tmp_value = 0
            for bias in biases:
                tmp_value += bias[tag][h]
                # value = bias[tag][h]
                # if value < tmp_value:
                #     tmp_value = bias[tag][h]
                tmp_value = tmp_value / len(biases)
            ave_h1_single.append(tmp_value)
        biases_dict[tag] = tf.Variable(ave_h1_single)

    return weights_dict, biases_dict


def append_large_gradient(sess, g, X, logits, formu, train_set_X, train_set_Y, catagory, to_be_appended_gradient_points_number,
                          decision_combination):
    new_train_set_X = []
    new_train_set_Y = []

    gradientList = g[0].tolist()

    gradient_size_list = []

    dimension = len(train_set_X[0])
    input_size = len(train_set_X)

    # print (type(gradientList))
    for i in range(input_size):
        size = util.calculate_vector_size(g[0][i])
        gradient_size_list.append(size)

    gradient_size_list.sort()

    index = -to_be_appended_gradient_points_number
    if(to_be_appended_gradient_points_number > len(gradient_size_list)):
        index = 0

    size_threshold = gradient_size_list[index]
    moving_step = util.calculate_std_dev(train_set_X)
    count = 0
    for j in range(input_size):
        size = util.calculate_vector_size(gradientList[j])
        if size < size_threshold:
            continue
        count += 1
        if count>to_be_appended_gradient_points_number:
            continue
        # value = sess.run(logits, feed_dict={X:[train_set_X[j]]})
        new = br.decide_cross_boundary_point(sess, g[0][j], size, X, logits,
                                             train_set_X[j], decision_combination, moving_step)

        # new_value = sess.run(logits, feed_dict={X:[new]})

        if (len(new) != 0):
            if (new not in train_set_X):
                new_train_set_X.append(new)

                flag = testing_function.test_label(new, formu)
                if (flag):
                    new_train_set_Y.append([1])
                else:
                    new_train_set_Y.append([0])

    train_set_X = train_set_X + new_train_set_X
    train_set_Y = train_set_Y + new_train_set_Y

    return train_set_X, train_set_Y


def generate_accuracy(train_path, test_path, formula, category, learning_rate, training_epochs, lower_bound, upper_bound, parts_num, usebagging):
    print("=========GRADIENT===========")

    # Parameters
    # learning_rate = 0.1
    # training_epochs = 100
    balance_ratio_threshold = 0.7
    active_learning_iteration = 5

    to_be_appended_random_points_number = 6
    to_be_appended_gradient_points_number = 6
    to_be_appended_boundary_remaining_points_number = 6

    train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_path, test_path, read_next=True)
    net_stru = ns.NNStructure(train_set_X[0], learning_rate)

    newgrads = tf.gradients(net_stru.logits, net_stru.X)

    y = None

    train_acc_list = []
    test_acc_list = []
    result = []
    train_acc_max = 0

    decision = gradient_combination.combination(len(train_set_X[0]))
    save_path = "model_saved/gradient_model"
    # predicted = tf.cast(net_stru.logits > 0, dtype=tf.float32)
    saver = tf.train.Saver()
    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))
        # ten times training
        with tf.Session() as sess:
            sess.run(net_stru.init)
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            print(len(label_0), len(label_1))
            if (len(label_1) == 0 or len(label_0) == 0):
                raise Exception("Cannot be classified")
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            # if (not util.is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
            #     br.apply_boundary_remaining(sess, newgrads, net_stru.X, net_stru.Y, length_0, length_1, net_stru.logits,
            #                                 formula,
            #                                 train_set_X, train_set_Y, to_be_appended_boundary_remaining_points_number)
                # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
                #                         train_set_Y, 10+i)
            all_data_X, all_data_Y = partition_data(label_0, label_1, parts_num)
            # print(all_data_X, all_data_Y)

            all_weights_dict = []
            all_biases_dict = []
            all_weights = {}
            all_biases = {}
            if usebagging:
                for parts in range(parts_num):
                    best_accuracy = 0
                    sess.run(net_stru.init)
                    for epoch in range(training_epochs):
                        _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                        feed_dict={net_stru.X: all_data_X[parts], net_stru.Y: all_data_Y[parts]})
                        train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
                        train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
                        if train_acc > best_accuracy:
                            best_accuracy = train_acc
                            # print("best_accuracy ", best_accuracy)
                            saver.save(sess, './models/benchmark.ckpt')

                    saver.restore(sess, "./models/benchmark.ckpt")
                    # predicted = tf.cast(net_stru.logits > 0.5, dtype=tf.float32)
                    # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}),
                    #                             train_set_X, train_set_Y,
                    #                             lower_bound, upper_bound, -1)

                    weights_dict = sess.run(net_stru.weights)
                    bias_dict = sess.run(net_stru.biases)

                    all_weights_dict.append(weights_dict)
                    all_biases_dict.append(bias_dict)

                # initialize dictionary
                for key in all_weights_dict[0].keys():
                    all_weights[key]  = []
                    for dim in range(len(all_weights_dict[0][key])):
                        if key != "out":
                            all_weights[key].append([])
                for key in all_biases_dict[0].keys():
                    all_biases[key]  = []

                # combine all weights and biases
                for key in all_weights_dict[0].keys():
                    if key == "out":
                        for cnt in range(len(all_weights_dict)):
                            for dim in range(len(all_weights_dict[0][key])):
                                weights_list = [k/parts_num for k in all_weights_dict[cnt][key][dim]]
                                all_weights[key].append(weights_list)
                    else:
                        for cnt in range(len(all_weights_dict)):   
                            for dim in range(len(all_weights_dict[0][key])):
                                weights_list = [k for k in all_weights_dict[cnt][key][dim]]
                                all_weights[key][dim] += weights_list
                for key in all_biases_dict[0].keys():
                    if key == "out":
                        for cnt in range(len(all_biases_dict)):
                            all_biases[key].append(all_biases_dict[cnt][key][0])
                    else:
                        for cnt in range(len(all_biases_dict)):
                            bias_list = [k for k in all_biases_dict[cnt][key]]
                            all_biases[key] += bias_list
                
                total = 0
                for out_bias in all_biases["out"]:
                    total += out_bias
                all_biases["out"] = [total/parts_num]
                
                for key in all_weights.keys():
                    # print(all_weights[key])
                    all_weights[key] = tf.Variable(all_weights[key])
                for key in all_biases.keys():
                    all_biases[key] = tf.Variable(all_biases[key])

                # print(all_biases_dict)
                # weight_tags = list(all_weights_dict[0].keys())
                # bias_tag = list(all_biases_dict[0].keys())

                # weights, biases = calculate_average(all_weights_dict, all_biases_dict, weight_tags, bias_tag)

                # print("weights average:", sess.run(weights))
                # print("bias average:", type(biases["b1"]))

                # print(len(weights["out"]), len(biases["out"]))
                #
                # print(type(all_weights_dict[0]["h1"]))
                net_stru_ = ns.NNStructureFixedVar(train_set_X[0], learning_rate, all_weights, all_biases)
                # with tf.Session as session:
                #     session.run(net_stru_.init)
            else:
                for epoch in range(training_epochs):
                    _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                    feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})
                net_stru_ = net_stru

                train_y = sess.run(net_stru_.logits, feed_dict={net_stru_.X: train_set_X})
                train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
                if train_acc > best_accuracy:
                    best_accuracy = train_acc
                    # print("best_accuracy ", best_accuracy)
                    saver.save(sess, './models/benchmark.ckpt')

            # saver.restore(sess, "./models/benchmark.ckpt")
            sess.run(net_stru_.init)

            train_y = sess.run(net_stru_.logits, feed_dict={net_stru_.X: train_set_X})
            test_y = sess.run(net_stru_.logits, feed_dict={net_stru_.X: test_set_X})

            print("Bagging performance")
            train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            test_acc = util.calculate_accuracy(test_y, test_set_Y, False)

            print(train_acc)
            print(test_acc)

            # saver = tf.train.Saver()
            # save model
            if len(train_acc_list) == 0:
                saver.save(sess, save_path)
                train_acc_max = train_acc
            else:
                if train_acc >= train_acc_max:
                    print("Got better result")
                    saver.save(sess, save_path)
                    train_acc_max = train_acc
                else:
                    print("not a better result")

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            # if i == 2:
            #     with open("test.csv", "w") as file:
            #         wr = csv.writer(file, dialect='excel')
            #         for line in range(len(train_set_X)):
            #             tmp = [float(train_set_Y[line][0])] + train_set_X[line]
            #             wr.writerow(tmp)
            predicted = tf.cast(net_stru_.logits > 0, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru_.X: x}), train_set_X,
                                        train_set_Y, lower_bound, upper_bound, i)

            g = sess.run(newgrads, feed_dict={net_stru.X: train_set_X})
            # print(g)

            train_set_X, train_set_Y = append_large_gradient(sess, g, net_stru_.X, net_stru_.logits, formula, train_set_X,
                                                             train_set_Y, category, to_be_appended_gradient_points_number, decision)
            #
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, 20+i)

            train_set_X, train_set_Y = util.append_random_points(formula, train_set_X, train_set_Y,
                                                                 to_be_appended_random_points_number, lower_bound,
                                                                 upper_bound)
            # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
            #                             train_set_Y, 30+i)

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

    ###########################################
    # decide new points dimension by dimension
    # for j in range(len(train_set_X)):
    #     grad = 0
    #     for k in range(n_input):
    #         grad += g[0][j][k] * g[0][j][k]
    #     g_total = math.sqrt(grad)
    #     # print("Im here ==================================")
    #     new = []
    #     if (g_total > threshold):
    #         for k in range(n_input):

    #             tmp1 = [x for x in train_set_X[j]]
    #             tmp1[k] = tmp1[k] + g[0][j][k] * (step / g_total)
    #             tmp2 = [x for x in train_set_X[j]]
    #             tmp2[k] = tmp2[k] - g[0][j][k] * (step / g_total)

    #             new_pointsX = [tmp1, tmp2, train_set_X[j]]
    #             new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})

    #             original_y = new_pointsY[-1]
    #             distances = [x for x in new_pointsY]
    #             distances = distances[:-1]
    #             # ans = 0
    #             if (original_y < 0.5):
    #                 ans = max(distances)
    #             else:
    #                 ans = min(distances)
    #             one_position = new_pointsX[distances.index(ans)]
    #             if (one_position==tmp1):
    #                 new.append(tmp1[k])
    #             else:
    #                 new.append(tmp2[k])

    #############################################

    ##boundary remaining test
    ##small gradient test
    #         X1=train_set_X[j][0]
    #         X2=train_set_X[j][1]
    #         newY=train_set_Y[j][0]
    #         g_x = g[0][j][0]
    #         g_y = g[0][j][1]
    #         g_total = math.sqrt(g_x*g_x+g_y*g_y)

    #         if (g_total==0) :
    #             tmpX1 = X1 - step
    #             tmpX2 = X2 + step
    #         else:
    #             tmpX1 = X1 - g[0][j][1]*(step/g_total)
    #             tmpX2 = X2 + g[0][j][0]*(step/g_total)
    #         ##print ("Y",newY)
    #         if(g[0][j][0]<0.01):

    #         	smallGradient_total+=1
    #         	if(newY==0):
    #         		if(polynomial_model(tmpX1,tmpX2)):
    #         			smallGradient_Unchanged+=1.0
    #         	elif(newY==1):
    #         		if(not polynomial_model(tmpX1,tmpX2)):
    #         			smallGradient_Unchanged+=1.0

    #         # ##large gradient test
    #         if(g[0][j][0]>0.01):
    #         	# newtmpX1=train_set_X[j][0]-g[0][j][0]*k
    #         	# newtmpX2=train_set_X[j][1]-g[0][j][1]*k

    #         	largeGradient_total+=1
    #         	if(newY==0):
    #         		if(polynomial_model(tmpX1,tmpX2)):
    #         			largeGradient_Unchanged+=1.0
    #         	elif(newY==1):
    #         		if(not polynomial_model(tmpX1,tmpX2)):
    #         			largeGradient_Unchanged+=1.0

    # # print("generated data points:")
    # for j in range(len(new_train_set_X)):
    #     print("(", new_train_set_X[j][0], ", ", new_train_set_X[j][1], ", ", new_train_set_X[j][2], ")", "label: ", new_train_set_Y[j][0])
    # if (smallGradient_total != 0) :
    #     print ("Small gradients", smallGradient_Unchanged/smallGradient_total)
    # if (largeGradient_total != 0):
    #     print ("Large gradients", largeGradient_Unchanged/largeGradient_total)
    # print(train_set_X)

    # print(smallGradient_total)
    # print (smallGradient_Unchanged)
    # print(largeGradient_total)
    # print (largeGradient_Unchanged)

    # print ("small gradient unchanged rate: ",smallGradient_Unchanged/smallGradient_total)
    # print ("large gradient unchanged rate: ", largeGradient_Unchanged/largeGradient_total)
    with tf.Session() as sess:
        saver.restore(sess, save_path)
        # util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}), train_set_X,
        #                             train_set_Y, lower_bound,upper_bound,-1)
        sess.run(net_stru_.init)
        print("weights final:", sess.run(net_stru_.weights))
        train_y = sess.run(net_stru_.logits, feed_dict={net_stru_.X: train_set_X})
        test_y = sess.run(net_stru_.logits, feed_dict={net_stru_.X: test_set_X})

        train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
        test_acc = util.calculate_accuracy(test_y, test_set_Y, False)
        print("Training: ", train_acc)
        print("Testing: ", test_acc)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    result.append(train_acc_list)
    result.append(test_acc_list)
    # tf.reset_default_graph()
    return result
