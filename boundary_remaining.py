# TODO move the code here

import math
import random

import gradient_combination
import testing_function
import util
import random

def initialize_processing_points(sess, new_grads, X, Y, length_0, length_1, train_set_X, train_set_Y):
    label_selected = []
    gradient_selected = []
    to_be_added_number = 0

    # compare if data is unbalanced
    label_flag = 0
    g = sess.run(new_grads, feed_dict={X: train_set_X})
    # print(train_set_X)
    # print ("grad",g[0])
    label_0, label_1, label_0_gradient, label_1_gradient \
        = util.data_partition_gradient(train_set_X, train_set_Y, g[0])

    if length_0 < length_1:
        label_selected = label_0
        gradient_selected = label_0_gradient
        to_be_added_number = length_1 - length_0
        label_flag = 0
    else:
        label_selected = label_1
        gradient_selected = label_1_gradient
        to_be_added_number = length_0 - length_1
        label_flag = 1
    print("label selected: ", label_selected)
    return label_selected, gradient_selected, to_be_added_number


def apply_boundary_remaining(sess, new_grads, X, Y, length_0, length_1, logits, formu, train_set_X, train_set_Y):
    points, preliminary_gradients, to_be_added_number = \
        initialize_processing_points(sess, new_grads, X, Y, length_0, length_1, train_set_X, train_set_Y)

    ################################################################
    # get all gradients for the unbalanced label points

    # boundary remaining
    # print(g)
    gradients = decide_all_gradients_for_boundary_remaining(
        X, preliminary_gradients, points, logits, sess)

    std_dev = util.calculate_std_dev(points, train_set_X)
    print("standard deviation", std_dev)

    bias_direction = length_0 > length_1
    newX = balancing_points(bias_direction, points, gradients, to_be_added_number, formu, std_dev)

    for k in newX:
        label = testing_function.test_label(k, formu)
        print(k, label)
        if (label):
            train_set_X.append(k)
            train_set_Y.append([1])
        else:
            train_set_X.append(k)
            train_set_Y.append([0])

    print("new training size after boundary remaining", len(train_set_X), len(train_set_Y))


def decide_all_gradients_for_boundary_remaining(X, gradient_selected, label_selected, logits, sess):
    gradient_list = []
    decision_options = gradient_combination.combination(len(label_selected[0]))
    for j in range(len(label_selected)):
        grad = 0
        dimension = len(label_selected[0])
        for k in range(dimension):
            grad += gradient_selected[j][k] * gradient_selected[j][k]
        gradient_length = math.sqrt(grad)

        # select all-1 direction if the gradient is all-0
        if gradient_length == 0:
            tmpg = []
            for d in range(dimension):
                randomPower=random.randint(1,2)
                sigh=(-1)**randomPower
                randomNumber=(random.randint(1, 10))*sigh
                tmpg.append(randomNumber)
            gradient_list.append(tmpg)
            print("random direction",tmpg)
            continue

        #############################################################

        # TODO decision_direction should return a direction towards boudary
        direction = decision_direction(X, decision_options, gradient_length,
                                       gradient_selected, j, label_selected, logits, sess,inverse=True)

        # TODO calculate your own direction based on the above direction

        ####################################################
        # direction = decide_direction_2n0(X, gradient_length, j, label_selected, logits, sess, step, train_set_X)
        # continue working on this part
        #######################################################

        return_value = []
        for k in range(len(direction)):
            if direction[k] == True:
                return_value.append(-gradient_selected[j][k])
            else:
                return_value.append(gradient_selected[j][k])
        gradient_list.append(return_value)
    return gradient_list


# def decide_direction_2n0(X, gradient_length, j, label_selected, logits, sess, step, train_set_X):
#     new = []
#     n_input = len(label_selected[0])
#     for k in range(n_input):
#         tmp1 = []
#         tmp2 = []
#         for h in range(n_input):
#             if h == k:
#                 tmp1.append(train_set_X[j][h] - g[0][j][h] * (step / gradient_length))
#                 tmp2.append(train_set_X[j][h] + g[0][j][h] * (step / gradient_length))
#             else:
#                 tmp1.append(train_set_X[j][h])
#                 tmp2.append(train_set_X[j][h])
#
#         new_pointsX = [tmp1, tmp2, train_set_X[j]]
#         new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})
#
#         original_y = new_pointsY[-1]
#         distances = [x for x in new_pointsY]
#         distances = distances[:-1]
#         # ans = 0
#         if (original_y < 0.5):
#             ans = max(distances)
#         else:
#             ans = min(distances)
#         one_position = new_pointsX[distances.index(ans)]
#         if (one_position == tmp1):
#             new.append(tmp1[k])
#         else:
#             new.append(tmp2[k])
#     return n_input


def decision_direction(X, decision_options, gradient_length, gradient_selected, j, label_selected, logits, sess,inverse):
    step = 1

    new_pointsX = []
    for decision_option in decision_options:
        tmp = []
        for h in range(len(label_selected[0])):
            if (decision_option[h] == True):
                tmp.append(label_selected[j][h] - gradient_selected[j][h] * (step / gradient_length))
            else:
                tmp.append(label_selected[j][h] + gradient_selected[j][h] * (step / gradient_length))
        # tmp[k].append(train_set_X[j][k] + g[0][j][k] * (step / g_total))
        new_pointsX.append(tmp)
    new_pointsX.append(label_selected[j])
    new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})
    original_y = new_pointsY[-1]
    values = [x for x in new_pointsY]
    values = values[:-1]

    ans = 0
    # ans_gradient = 0
    if (original_y < 0.5):
        if inverse==True:
            ans = min(values)
        else:
            ans = max(values)
    else:
        if inverse==True:
            ans = max(values)
        else:
            ans = min(values)
    direction = decision_options[values.index(ans)]
    # point_gradient = new_pointsX[values.index(ans_gradient)]
    return direction


def balancing_points(inflag, points, gradient, length_added, formu, std_dev):
    print("inflag: ", inflag)
    times = 0
    outputX = []
    iter = 0
    flag = False
    count = 0.0
    wrong = 0.0
    step = std_dev + random.uniform(0, 1)
    while True:
        for i in range(len(points)):
            g_total = 0
            grad = 0
            for k in range(len(points[0])):
                grad += gradient[i][k] * gradient[i][k]
            g_total = math.sqrt(grad)
            tmpList = []

            for j in range(len(points[i])):
                tmpValue = points[i][j] + gradient[i][j] * (step / g_total)
                tmpList.append(tmpValue)

            point_label = testing_function.test_label(tmpList, formu)
            print (point_label)
            count += 1

            if inflag == True and point_label:
                times += 1
                wrong += 1
                if times > 100:
                    flag = True
                    break
                continue
            if inflag == False and not point_label:
                times += 1
                wrong += 1
                if times > 100:
                    flag = True
                    break
                continue
            outputX.append(tmpList)
            times += 1
            if times > 100:
                flag = True
                break
            iter += 1
            if (iter == length_added):
                flag = True
                break
        if (flag == True):
            break

    print("points added \n", outputX)
    print("Boundary remaining accuracy: ", float((count - wrong) / count))
    return outputX

# label_0=[[1,2]]
# label_1=[[1,2],[3,4],[5,6],[7,8],[9,10]]

# gra0=[[0.1,0.2]]
# gra1=[]
# balancing_points(label_0,label_1,gra0,gra1)

def decide_cross_boundry_point(sess, g, X, logits, train_set_X, j, threshold, decision):
    step = random.uniform(2, 4)
    grad = 0

    dimension = len(train_set_X[0])

    for k in range(dimension):
        grad += g[0][j][k] * g[0][j][k]
    g_total = math.sqrt(grad)
    # print("Im here ==================================")

    new = []
    if (g_total > threshold):
        new_pointsX = []
        for k in range(len(decision)):
            tmp = []
            for h in range(dimension):
                if (decision[k][h]==True):
                    tmp.append(train_set_X[j][h] - g[0][j][h] * (step / g_total))
                else:
                    tmp.append(train_set_X[j][h] + g[0][j][h] * (step / g_total))
            # tmp[k].append(train_set_X[j][k] + g[0][j][k] * (step / g_total))
            new_pointsX.append(tmp)
        new_pointsX.append(train_set_X[j])
        new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})

        original_y = new_pointsY[-1]
        distances = [x for x in new_pointsY]
        distances = distances[:-1]
        # ans = 0
        if (original_y < 0.5):
            ans = max(distances)
        else:
            ans = min(distances)
        new = new_pointsX[distances.index(ans)]
        # print("origin point: ", train_set_X[j])
        # print("new point: ", new)
    return new
