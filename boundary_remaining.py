# TODO move the code here

import math
import random

import gradient_combination
import testing_function
import util


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

    return label_selected, gradient_selected, to_be_added_number


def apply_boundary_remaining(sess, new_grads, X, Y, length_0, length_1,
                             logits, formu, train_set_X, train_set_Y, to_be_appended_boundary_remaining_points_number,
                             type, name_list, mock):
    points_in_less_side, preliminary_gradients_in_less_side, to_be_added_number = \
        initialize_processing_points(sess, new_grads, X, Y, length_0, length_1, train_set_X, train_set_Y)

    if to_be_added_number > to_be_appended_boundary_remaining_points_number:
        to_be_added_number = to_be_appended_boundary_remaining_points_number

    gradients = decide_all_gradients_for_boundary_remaining(
        X, preliminary_gradients_in_less_side, points_in_less_side, logits, sess)

    std_dev = util.calculate_std_dev(train_set_X)

    bias_direction = length_0 > length_1
    random.shuffle(points_in_less_side)
    print("less side points:", points_in_less_side)
    newX = balancing_points(bias_direction, points_in_less_side, gradients, to_be_added_number, formu, std_dev, type,
                            name_list, mock, logits)
    print("newX;", newX)
    flagList = testing_function.test_label(newX, formu, type, name_list, mock)
    for k in range(len(newX)):
        label = flagList[k]
        if (label):

            train_set_X.append(newX[k])
            train_set_Y.append([1])
            print("added point: ", newX[k], label)
        else:
            train_set_X.append(newX[k])
            train_set_Y.append([0])
            print("added point: ", newX[k], label)


def decide_all_gradients_for_boundary_remaining(X, gradient_selected, label_selected, probability, sess):
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
                randomPower = random.randint(1, 2)
                sigh = (-1) ** randomPower
                randomNumber = (random.randint(1, 10)) * sigh
                tmpg.append(randomNumber)
            gradient_list.append(tmpg)
            continue

        #############################################################

        # TODO decision_direction should return a direction towards boundary
        direction = decision_direction(X, decision_options, gradient_length,
                                       gradient_selected, j, label_selected, probability, sess, inverse=True)

        # TODO calculate your own direction based on the above direction

        ####################################################
        # direction = decide_direction_2n0(X, gradient_length, j, label_selected, logits, sess, step, train_set_X)
        # continue working on this part
        #######################################################

        return_value = []
        for k in range(len(direction)):
            if direction[k]:
                return_value.append(-gradient_selected[j][k])
            else:
                return_value.append(gradient_selected[j][k])
        # print("Standard direction: ", return_value)
        random_direction = []
        dimension = len(direction)

        for i in range(dimension - 1):
            # print("Random_value: ", random.uniform(-10, 10))
            random_value = random.uniform(-5, 5)
            random_direction.append(random_value)
        dot_product = 0
        for i in range(dimension - 1):
            dot_product += return_value[i] * random_direction[i]

        lower_bound = -dot_product / return_value[-1]
        last_direction = random.uniform(lower_bound, lower_bound + 10)

        random_direction.append(last_direction)
        # print("Random direction: ", random_direction)
        gradient_list.append(random_direction)
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


def decision_direction(X, decision_options, gradient_length, gradient_selected, j, label_selected, logits, sess,
                       inverse):
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
    if (original_y < 0):
        if inverse == True:
            ans = min(values)
        else:
            ans = max(values)
    else:
        if inverse == True:
            ans = max(values)
        else:
            ans = min(values)
    direction = decision_options[values.index(ans)]

    # point_gradient = new_pointsX[values.index(ans_gradient)]
    return direction


def balancing_points(is_label_1_side, points_in_less_side, gradients, length_added, formu, std_dev, type, name_list,
                     mock):
    add_points = []
    break_loop = False
    trial_count = 0.0
    wrong = 0.0
    step = random.uniform(std_dev / 2.0, std_dev)
    print("moving step: ", step)
    balancing_threshold = 100

    while True:
        for i in range(len(points_in_less_side)):
            print("step:", step)
            trial_count += 1
            if trial_count > balancing_threshold:
                break_loop = True
                break

            gradient_length = util.calculate_vector_size(gradients[i])
            tmp_point = []
            for j in range(len(points_in_less_side[i])):
                tmp_value = points_in_less_side[i][j] + gradients[i][j] * (step / gradient_length)
                tmp_point.append(tmp_value)
            input_point = []

            input_point.append(tmp_point)

            result = testing_function.test_label(input_point, formu, type, name_list, mock)
            point_label = None
            if result[0] == 0:
                point_label = False
            else:
                point_label = True
            if is_label_1_side and not point_label:
                wrong += 1
                tmp_step = step / 2.0
                trial_count, wrong, points_added, break_loop = handle_wrong_point(points_in_less_side[i], gradients[i],
                                                                                  tmp_step, trial_count, wrong,
                                                                                  is_label_1_side, formu,
                                                                                  balancing_threshold, tmp_point, type,
                                                                                  name_list, mock)
                add_points = add_points + points_added
                success = trial_count - wrong
                print("success", success)
                if (success >= length_added):
                    break_loop = True
                    break
                continue
            if not is_label_1_side and point_label:
                wrong += 1
                tmp_step = step / 2.0
                trial_count, wrong, points_added, break_loop = handle_wrong_point(points_in_less_side[i], gradients[i],
                                                                                  tmp_step, trial_count, wrong,
                                                                                  is_label_1_side, formu,
                                                                                  balancing_threshold, tmp_point, type,
                                                                                  name_list, mock)
                add_points = add_points + points_added
                success = trial_count - wrong
                print("success", success)
                if (success >= length_added):
                    break_loop = True
                    break
                continue
            add_points.append(tmp_point)

            success = trial_count - wrong
            print("success", success)
            if (success >= length_added):
                break_loop = True
                break

        if (break_loop):
            break

    print("Boundry remaining points added ", len(add_points))
    print("Boundary remaining accuracy: ", float((trial_count - wrong) / trial_count))
    for point in add_points:
        for index in range(len(point)):
            if type=="INTEGER":
                point[index]=int(round(point[index]))
    return add_points


# label_0=[[1,2]]
# label_1=[[1,2],[3,4],[5,6],[7,8],[9,10]]

# gra0=[[0.1,0.2]]
# gra1=[]
# balancing_points(label_0,label_1,gra0,gra1)

def decide_cross_boundary_point(sess, gradient_sample, gradient_size, X, probability, train_sample, decision_combination,
                                moving_step):
    # step = random.uniform(2, 4)
    grad = 0

    dimension = len(train_sample)

    new_pointsX = []
    for k in range(len(decision_combination)):
        tmp = []
        for h in range(dimension):
            if (decision_combination[k][h] == True):
                tmp.append(train_sample[h] - moving_step * (gradient_sample[h] / gradient_size))
            else:
                tmp.append(train_sample[h] + moving_step * (gradient_sample[h] / gradient_size))
        # tmp[k].append(train_set_X[j][k] + g[0][j][k] * (step / g_total))
        new_pointsX.append(tmp)

    new_pointsX.append(train_sample)

    # prediction = tf.sigmoid(logits)

    new_pointsY = sess.run(probability, feed_dict={X: new_pointsX})
    original_y = new_pointsY[-1]
    values = [x for x in new_pointsY]
    values = values[:-1]
    # ans = 0
    if (original_y < 0):
        ans = max(values)
    else:
        ans = min(values)
    new = new_pointsX[values.index(ans)]
    return new


def handle_wrong_point(point, gradient, step, trial_count, wrong, is_label_1_side, formu, balancing_threshold,
                       tmp_point, type, name_list, mock):
    print("handling wrong point")
    return_list = []
    correct_point = []
    wrong_point = []
    flag = False
    while True:
        if trial_count >= balancing_threshold:
            flag = True
            break
        gradient_length = util.calculate_vector_size(gradient)
        tmp_point = []
        for j in range(len(point)):
            direction_step = step / gradient_length
            tmp_value = point[j] + gradient[j] * direction_step

            tmp_point.append(tmp_value)
        input_point = []
        input_point.append(tmp_point)
        result = testing_function.test_label(input_point, formu, type, name_list, mock)
        point_label = None
        if result[0] == 0:
            point_label = False
        else:
            point_label = True
        trial_count += 1
        if is_label_1_side and not point_label:
            wrong += 1
            wrong_point = tmp_point
            step = step / 2.0
            continue
        if not is_label_1_side and point_label:
            wrong += 1
            wrong_point = tmp_point
            step = step / 2.0
            continue

        correct_point = tmp_point
        break
    if correct_point != []:
        return_list.append(correct_point)

    if wrong_point != []:
        return_list.append(wrong_point)
    print("one time handle wrong point added", return_list)
    return trial_count, wrong, return_list, flag

def added_balancing_points(is_label_1_side, points_in_less_side, gradients, length_added, formu, std_dev, type, name_list,
                     mock):
    add_points = []
    break_loop = False
    trial_count = 0.0
    wrong = 0.0
    step = random.uniform(std_dev / 2.0, std_dev)
    print("moving step: ", step)
    balancing_threshold = 100

    for i in range(len(points_in_less_side)):
        print("step:", step)
        trial_count += 1

        gradient_length = util.calculate_vector_size(gradients[i])
        tmp_point = []
        for j in range(len(points_in_less_side[i])):
            tmp_value = points_in_less_side[i][j] + gradients[i][j] * (step / gradient_length)
            tmp_point.append(tmp_value)
        input_point = []
        input_point.append(tmp_point)

        result = testing_function.test_label(input_point, formu, type, name_list, mock)
        point_label = None
        if result[0] == 0:
            point_label = False
        else:
            point_label = True
        
        add_points.append(tmp_point)

        success = trial_count - wrong
        print("success", success)

    print("Boundry remaining points added ", len(add_points))
    print("Boundary remaining accuracy: ", float((trial_count - wrong) / trial_count))
    for point in add_points:
        for index in range(len(point)):
            if type=="INTEGER":
                point[index]=int(round(point[index]))
    return add_points
