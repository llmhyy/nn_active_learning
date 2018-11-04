import csv
import math
import os
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from main import label_tester as lt
from prj_test import formula


def save_model(sess, model_folder, model_file):
    if model_folder is not None:
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)
    model_path = os.path.join(model_folder, model_file)
    saver = tf.train.Saver()
    p = saver.save(sess, model_path)
    return p


def move(point, direction, step):
    direction_len = calculate_vector_size(direction)
    new_point = []
    for j in range(len(point)):
        d = point[j] + direction[j] * (step / direction_len)
        new_point.append(d)

    return new_point


def reset_random_seed():
    random_seed = 100
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)


def direction_combination(n):
    output = []
    if n == 1:
        a = [True]
        b = [False]
        output.append(a)
        output.append(b)
        return output
    else:
        prev_binary_combination = direction_combination(n - 1)
        binary_combination = []
        for i in prev_binary_combination:
            a = i + [True]
            binary_combination.append(a)
            b = i + [False]
            binary_combination.append(b)
        return binary_combination


def plot_clustering_result(clusters, lower_bound, upper_bound, iteration):
    train_set_X = []
    train_set_Y = []

    for i in range(len(clusters)):
        cluster = clusters[i]
        train_set_X = train_set_X + cluster
        for j in range(len(clusters[i])):
            train_set_Y.append(i + 1)

    X = np.array(train_set_X)
    Y = np.array(train_set_Y)
    y = Y.reshape(len(Y))
    # colors = cm.rainbow(np.linspace(0, 1, len(clusters)))

    plt.clf()
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(lower_bound, upper_bound)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    # plt.show()
    file_name = 'picture/clustering' + str(iteration) + '.png'
    plt.savefig(file_name)
    pass


def plot_decision_boundary(pred_func, train_set_X, train_set_Y, lower_bound, upper_bound, iteration):
    # Set min and max values and give it some padding
    x_min = lower_bound
    y_min = lower_bound

    x_max = upper_bound
    y_max = upper_bound

    X = np.array(train_set_X)
    Y = np.array(train_set_Y)

    h = 10
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    kx = xx.ravel()
    ky = yy.ravel()

    list = []
    for i in range(len(kx)):
        list.append([kx[i], ky[i]])

    Z = pred_func(list)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.copper)
    y = Y.reshape(len(Y))
    # plt.clf()
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(lower_bound, upper_bound)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    # plt.show()
    file_name = 'picture/test' + str(iteration) + '.png'
    plt.savefig(file_name)
    pass


# def plot_decision_boundary(pred_func, train_set_X, train_set_Y, iteration):
#     X = np.array(train_set_X)
#     Y = np.array(train_set_Y)

#     # Set min and max values and give it some padding
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
#     h = 10
#     # Generate a grid of points with distance h between them
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     # Predict the function value for the whole gid
#     kx = xx.ravel()
#     ky = yy.ravel()

#     list = []
#     for i in range(len(kx)):
#         list.append([kx[i], ky[i]])

#     Z = pred_func(list)
#     Z = Z.reshape(xx.shape)
#     # Plot the contour and training examples
#     plt.contourf(xx, yy, Z, cmap=plt.cm.copper)
# y = Y.reshape(len(Y))
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
# plt.show()
# file_name = 'test' + str(iteration) + '.png'
# plt.savefig(file_name)


def calculate_accuracy(y, set_y, print_data_details):
    test_correct = []
    test_wrong = []
    train_correct = []
    train_wrong = []
    for i in range(len(set_y)):
        # if (print_data_details):
        #     print(i, " predict:", y[i][0], " actual: ", set_Y[i][0])
        if math.isnan(y[i][0]):
            print(i, " predict:", y[i][0], " actual: ", set_y[i][0])

        if y[i][0] > 0.5 and set_y[i][0] == 1:
            test_correct.append(y[i])
        elif y[i][0] > 0.5 and set_y[i][0] == 0:
            test_wrong.append(y[i])
        elif y[i][0] <= 0.5 and set_y[i][0] == 0:
            test_correct.append(y[i])
        else:
            test_wrong.append(y[i])
    # print (test_correct)
    # print (test_wrong)
    result = len(test_correct) / float(len(test_correct) + len(test_wrong))
    # print(result)
    return result


def read_data_from_file(train_path, test_path, read_next):
    test_set_x = []
    test_set_y = []
    train_set_x = []
    train_set_y = []

    # # read training data
    # with open(train_path, 'r+', newline='') as csvfile:
    #     with open('./dataset/train_next.csv', 'w', newline='') as file:
    #         i = 0
    #         spamreader = csv.reader(csvfile)
    #         writer = csv.writer(file)
    #         for row in spamreader:
    #             if (i < 0 or i > 200):
    #                 i += 1
    #                 continue
    #             else:
    #                 i += 1
    #                 writer.writerow(row)
    #     file.close()

    # read testing data
    # test_path = "./dataset/test[-1]_[-1]_[2, -4, -3, 5]_[-1]_[-4, -2, 3]_[4, 0, -5]_[3, 5]_[2, 1, -1]_[2, -1]_8624.csv"
    with open(test_path, 'r+', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if len(row) == 0:
                continue

            # print(row)
            l = [float(x) for x in row]
            # print(l)
            test_set_x.append(l[1:])
            if row[0] == '1.0':
                test_set_y.append([1])
            else:
                test_set_y.append([0])

    # read training data
    # if read_next:
    #     train_path = './dataset/train_next.csv'
    with open(train_path, 'r+', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            if (len(row) == 0):
                continue
            l = [float(x) for x in row]
            # print(l)
            train_set_x.append(l[1:])
            if row[0] == '1.0':
                train_set_y.append([1])
            else:
                train_set_y.append([0])

    return train_set_x, train_set_y, test_set_x, test_set_y


def calculate_vector_size(vector):
    vec = np.array(vector)
    vec_length = np.linalg.norm(vec)
    return vec_length


def quickSort(alist):
    quickSortHelper(alist, 0, len(alist) - 1)


def quickSortHelper(alist, first, last):
    if first < last:
        splitpoint = partition(alist, first, last)

        quickSortHelper(alist, first, splitpoint - 1)
        quickSortHelper(alist, splitpoint + 1, last)


def partition(alist, first, last):
    pivot_value = alist[first]

    left_mark = first + 1
    right_mark = last

    done = False
    while not done:

        while left_mark <= right_mark and alist[left_mark] <= pivot_value:
            left_mark = left_mark + 1

        while alist[right_mark] >= pivot_value and right_mark >= left_mark:
            right_mark = right_mark - 1

        if right_mark < left_mark:
            done = True
        else:
            temp = alist[left_mark]
            alist[left_mark] = alist[right_mark]
            alist[right_mark] = temp

    temp = alist[first]
    alist[first] = alist[right_mark]
    alist[right_mark] = temp

    return right_mark


def calculate_distance(m, n):
    distance = 0
    for d in range(len(m)):
        distance += (m[d] - n[d]) * (m[d] - n[d])
    distance = math.sqrt(distance)
    return distance


def calculate_std_dev(train_set_x):
    if len(train_set_x) == 1:
        return np.random.uniform(1, 10)

    dimension = len(train_set_x[0])
    point_distance_list = []
    for p in range(len(train_set_x) - 1):
        for q in range(p + 1, len(train_set_x)):
            distance = 0
            for d in range(dimension):
                distance += (train_set_x[p][d] - train_set_x[q][d]) * (train_set_x[p][d] - train_set_x[q][d])
            distance = math.sqrt(distance)
            point_distance_list.append(distance)
    std_dev = np.std(point_distance_list)
    return std_dev


def data_partition(train_set_X, train_set_Y):
    label_0 = []
    label_1 = []
    for i in range(len(train_set_X)):
        if (train_set_Y[i][0] == 0):
            label_0.append(train_set_X[i])
        elif (train_set_Y[i][0] == 1):
            label_1.append(train_set_X[i])
    return label_0, label_1


def add_distance_values(number, distance_list, selected_list, pointer):
    pivot = 0
    while pivot < number:
        if distance_list[pointer] in selected_list:
            pointer += 1
        # add large points
        else:
            selected_list.append(distance_list[pointer])
            pivot += 1
            pointer += 1


def data_partition_gradient(train_set_X, train_set_Y, gradient):
    label_0 = []
    label_0_gradient = []
    label_1 = []
    label_1_gradient = []
    for i in range(len(train_set_X)):
        if (train_set_Y[i][0] == 0):
            label_0.append(train_set_X[i])
            label_0_gradient.append(gradient[i])
        elif (train_set_Y[i][0] == 1):
            label_1.append(train_set_X[i])
            label_1_gradient.append(gradient[i])
    return label_0, label_1, label_0_gradient, label_1_gradient


def append_random_points(formu, train_set_x, train_set_y, to_be_appended_random_points_number, lower_bound, upper_bound,
                         type, name_list, mock):
    category = formu.get_category()
    if category == formula.POLYNOMIAL:
        new_points_x, new_points_y = generate_polynomial_points(formu, to_be_appended_random_points_number, lower_bound,
                                                                upper_bound)
        train_set_x = train_set_x + new_points_x
        train_set_y = train_set_y + new_points_y
    elif category == formula.POLYHEDRON:
        new_points_x, new_points_y = generate_polyhedron_points(formu, to_be_appended_random_points_number, lower_bound,
                                                                upper_bound)
        train_set_x = train_set_x + new_points_x
        train_set_y = train_set_y + new_points_y
    print("New random points X", new_points_x)
    print("New random points Y", new_points_y)
    return train_set_x, train_set_y


def is_training_data_balanced(length_0, length_1, balance_ratio_threshold):
    return (length_0 / length_1 > balance_ratio_threshold and length_0 / length_1 <= 1) \
           or \
           (length_1 / length_0 > balance_ratio_threshold and length_1 / length_0 <= 1)


def generate_polynomial_points(single_formula, to_be_appended_random_points_number, lower_bound, upper_bound):
    single_formula = single_formula.get_formula()
    coefficientList = single_formula[:-1]
    y = single_formula[-1]
    outputX = []
    outputY = []
    for i in range(to_be_appended_random_points_number):
        xList = []
        variableNum = len(coefficientList)
        for j in range(variableNum):
            xList.append(random.randint(lower_bound, upper_bound))

        formula_tester = lt.FormulaLabelTester(single_formula)
        flag = formula_tester.polynomial_model(coefficientList, xList, y)
        outputX.append(xList)

        if (flag):
            outputY.append([1])
        else:
            outputY.append([0])

    return outputX, outputY


# TODO use lower and upper bound to generate data points
def generate_polyhedron_points(formu, to_be_appended_random_points_number, lower_bound, upper_bound):
    formu = formu.get_formula()
    dim = len(formu[0][0])
    outputX = []
    outputY = []

    for i in range(to_be_appended_random_points_number):
        generated_point = []
        for j in range(dim):
            generated_point.append(random.uniform(lower_bound, upper_bound))

        # k = random.randint(2, 3)
        # if k % 2 == 0:
        #     center = random.randint(0, len(formu[0]) - 1)
        #     for i in range(dim):
        #         generated_point.append(random.uniform(int(formu[0][center][i]) - 10, int(formu[0][center][i]) + 10))
        # else:
        #     for i in range(dim):
        #         generated_point.append(random.uniform(-10, 10))
        formula_tester = lt.FormulaLabelTester()
        flag = formula_tester.polycircle_model(formu[0], formu[1], generated_point)
        outputX.append(generated_point)

        if (flag):
            outputY.append([1])
        else:
            outputY.append([0])
    return outputX, outputY


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    np.seterr(all='raise')
    return vector / np.linalg.norm(vector)


def calculate_vector_angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculate_orthogonal_direction(vector):
    length = len(vector)
    v = vector[0: length - 1]
    r = np.random.randn(length - 1)

    v_a = np.array(v)
    v_r = np.array(r)
    value = np.dot(v_a, v_r)

    last_element = 0
    if vector[length - 1] != 0:
        last_element = -value / vector[length - 1]
        last_element = np.asscalar(last_element)

    direction = r.tolist()
    direction.append(last_element)

    return direction


def calculate_vector_projection(vector, gradient):
    '''
    project vector on the plane orthogonal to gradient
    '''
    vec = np.array(vector)
    vgra = np.array(gradient)

    gradient_length = np.linalg.norm(vgra)
    orthogonal_part = vgra / gradient_length * (np.dot(vec, vgra) / gradient_length)

    result = np.subtract(vec, orthogonal_part).tolist()
    return result


def calculate_direction(point, origin):
    tmp_vector = []
    for m in range(len(point)):
        new_value = point[m] - origin[m]
        tmp_vector.append(new_value)
    return tmp_vector
