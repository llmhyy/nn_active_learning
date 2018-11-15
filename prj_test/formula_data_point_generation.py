import csv
import math
import random
from itertools import product

import numpy as np

from main import label_tester as lt
from prj_test import formula


def generate_partitioned_data(single_formula, category, lower_bound, upper_bound, positive_num, negative_num):
    test_point_number = 100000

    train_positive_list = []
    train_negative_list = []

    test_positive_list = []
    test_negative_list = []

    # dim = 0
    if category == formula.POLYHEDRON:
        train_positive_list, train_negative_list = generate_partitioned_random_points_for_sphere(single_formula,
                                                                                                 lower_bound,
                                                                                                 upper_bound,
                                                                                                 positive_num,
                                                                                                 negative_num)
        test_positive_list, test_negative_list = generate_partitioned_random_points_for_sphere(single_formula,
                                                                                               lower_bound,
                                                                                               upper_bound,
                                                                                               test_point_number,
                                                                                               test_point_number)

        # dim = len(single_formula.get_formula()[0][0])
    elif category == formula.POLYNOMIAL:
        train_positive_list, train_negative_list = generate_partitioned_random_points_for_polynomial(single_formula,
                                                                                                     lower_bound,
                                                                                                     upper_bound,
                                                                                                     positive_num,
                                                                                                     negative_num)
        test_positive_list, test_negative_list = generate_partitioned_random_points_for_polynomial(single_formula,
                                                                                                     lower_bound,
                                                                                                     upper_bound,
                                                                                                   test_point_number,
                                                                                                   4*test_point_number)
        # dim = len(single_formula.get_formula()[:-1])

    train_set_x = train_positive_list + train_negative_list
    train_set_y_1 = np.ones((len(train_positive_list), 1)).tolist()
    train_set_y_0 = np.zeros((len(train_negative_list), 1)).tolist()
    train_set_y = train_set_y_1
    for y_0 in train_set_y_0:
        train_set_y.append(y_0)

    tmp = list(zip(train_set_x, train_set_y))
    random.shuffle(tmp)
    train_set_x, train_set_y = zip(*tmp)
    train_set_x = list(train_set_x)
    train_set_y = list(train_set_y)

    test_set_x = test_positive_list + test_negative_list
    test_set_y_1 = np.ones((len(test_positive_list), 1)).tolist()
    test_set_y_0 = np.zeros((len(test_negative_list), 1)).tolist()
    test_set_y = test_set_y_1
    for y_0 in test_set_y_0:
        test_set_y.append(y_0)

    return train_set_x, train_set_y, test_set_x, test_set_y


def generate_data(single_formula, category, lower_bound, upper_bound, data_num):
    dim = 0
    if category == formula.POLYHEDRON:
        train_set_x, train_set_y = generate_random_points_for_sphere(single_formula, lower_bound, upper_bound,
                                                                     data_num)
        dim = len(single_formula.get_formula()[0][0])
    elif category == formula.POLYNOMIAL:
        train_set_x, train_set_y = generate_partitioned_random_points_for_polynomial(single_formula, lower_bound,
                                                                                     upper_bound,
                                                                                     data_num)
        dim = len(single_formula.get_formula()[:-1])

    test_set_x, test_set_y = generate_testing_point(single_formula, category, dim, data_num,
                                                    lower_bound,
                                                    upper_bound)
    return train_set_x, train_set_y, test_set_x, test_set_y


def generate_data_with_file(single_formula, category, lower_bound, upper_bound, num):
    train_name = "train" + "_".join(str(x) for x in single_formula.get_formula()) + ".csv"
    test_name = "test" + "_".join(str(x) for x in single_formula.get_formula()) + ".csv"
    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    train_set_x, train_set_y = generate_data(single_formula, category, lower_bound, upper_bound, num)
    write_to_file(train_set_x, train_set_y, train_path)

    dim = 0
    if category == formula.POLYHEDRON:
        dim = len(coefficient_list=single_formula.get_formula()[:-1])
    elif category == formula.POLYNOMIAL:
        dim = len(single_formula.get_formula()[0][0])
    test_data_list, test_label_list = generate_testing_point(single_formula, category, dim, num, lower_bound,
                                                             upper_bound)
    write_to_file(test_data_list, test_label_list, test_path)

    return train_path, test_path


def generate_partitioned_data_with_file(single_formula, category, lower_bound, upper_bound, positive_num, negative_num):
    train_name = "train" + "_".join(str(x) for x in single_formula.get_formula()) + ".csv"
    test_name = "test" + "_".join(str(x) for x in single_formula.get_formula()) + ".csv"
    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    train_set_x, train_set_y = generate_partitioned_data(single_formula, category, lower_bound, upper_bound,
                                                         positive_num, negative_num)
    tmp = list(zip(train_set_x, train_set_y))
    random.shuffle(tmp)
    train_set_x, train_set_y = zip(*tmp)
    train_set_x = list(train_set_x)
    train_set_y = list(train_set_y)

    write_to_file(train_set_x, train_set_y, train_path)

    dim = 0
    if category == formula.POLYHEDRON:
        dim = len(coefficient_list=single_formula.get_formula()[:-1])
    elif category == formula.POLYNOMIAL:
        dim = len(single_formula.get_formula()[0][0])

    num = positive_num + negative_num
    test_data_list, test_label_list = generate_testing_point(single_formula, category, dim, num, lower_bound,
                                                             upper_bound)
    write_to_file(test_data_list, test_label_list, test_path)

    return train_path, test_path


def generate_partitioned_random_points_for_polynomial(single_formula, lower_bound, upper_bound, positive_num,
                                                      negative_num):
    formula_tester = lt.FormulaLabelTester(single_formula)
    positive_list = []
    negative_list = []

    coefficient_list = single_formula.get_formula()[:-1]
    y = single_formula.get_formula()[-1]
    while len(positive_list) < positive_num or len(negative_list) < negative_num:
        point = []
        variable_num = len(coefficient_list)
        for i in range(variable_num):
            point.append(random.randint(lower_bound, upper_bound))

        label = formula_tester.polynomial_model(coefficient_list, point, y)

        if label:
            if len(positive_list) < positive_num:
                positive_list.append(point)
        else:
            if len(negative_list) < negative_num:
                negative_list.append(point)

    return positive_list, negative_list


def generate_random_points_for_polynomial(single_formula, lower_bound, upper_bound, num):
    formula_tester = lt.FormulaLabelTester(single_formula)
    data_list = []
    label_list = []

    coefficient_list = single_formula.get_formula()[:-1]
    y = single_formula.get_formula()[-1]
    while len(data_list) < num:
        point = []
        variable_num = len(coefficient_list)
        for i in range(variable_num):
            point.append(random.randint(lower_bound, upper_bound))

        label = formula_tester.polynomial_model(coefficient_list, point, y)
        data_list.append(point)
        label_list.append([label])

    return data_list, label_list


def read_from_file(path):
    train_set_x = []
    train_set_y = []

    with open(path, 'rt') as f:
        reader = csv.reader(f)

        for row in reader:
            label = float(row[0][0])
            train_set_y.append(label)

            point = []
            for i in range(len(row) - 1):
                point.append(float(row[0][i + 1]))
            train_set_x.append(point)

    return train_set_x, train_set_y


def write_to_file(positive_list, negative_list, path):
    with open(path, 'w', newline='') as csv_file:
        train = csv.writer(csv_file)
        for point in positive_list:
            opt_list = [1.0]
            opt_list += point
            train.writerow(opt_list)
        for point in negative_list:
            opt_list = [0.0]
            opt_list += point
            train.writerow(opt_list)


def write_to_file(data_list, label_list, path):
    with open(path, 'w', newline='') as csv_file:
        train = csv.writer(csv_file)
        for i in range(len(data_list)):
            point = data_list[i]
            label = label_list[i][0]
            opt_list = [label]
            opt_list += point
            train.writerow(opt_list)


# generate random data points for a circle formula
def generate_random_points_for_sphere(single_formula, lower_bound, upper_bound,
                                      data_point_number):  # [[[12,0],[-12,0]],[4,4]]
    formu_list = single_formula.get_formula()
    dim = len(formu_list[0][0])

    data_list = []
    label_list = []
    for k in range(data_point_number):
        data_point = []
        point = []
        if k % 3 == 0:
            center = random.randint(0, len(formu_list[0]) - 1)
            for i in range(dim):
                point.append(
                    random.uniform(int(formu_list[0][center][i]) - 300, int(formu_list[0][center][i]) + 300))
        else:
            for i in range(dim):
                point.append(random.uniform(lower_bound, upper_bound))

        data_list.append(point)
        formula_tester = lt.FormulaLabelTester(single_formula)
        label = formula_tester.polycircle_model(formu_list[0], formu_list[1], point)
        label_list.append([label])
    return data_list, label_list


# generate random data points for a circle formula
def generate_partitioned_random_points_for_sphere(single_formula, lower_bound, upper_bound,
                                                  positive_num, negative_num):  # [[[12,0],[-12,0]],[4,4]]
    formu_list = single_formula.get_formula()
    dim = len(formu_list[0][0])

    positive_list = []
    negative_list = []
    while len(positive_list) < positive_num or len(negative_list) < negative_num:
        data_point = []

        center = random.choice(formu_list[0])
        index = formu_list[0].index(center)
        radius = formu_list[1][index]
        move = math.sqrt(radius ** 2 / dim)

        r = random.uniform(0, 1)

        point = []
        for i in range(dim):
            length = random.uniform(0, move)
            if r > 0.5:
                step = length
                value = center[i] + step
            else:
                value = random.uniform(lower_bound, upper_bound)
            point.append(value)

        formula_tester = lt.FormulaLabelTester(single_formula)
        label = formula_tester.polycircle_model(formu_list[0], formu_list[1], point)

        if label:
            if len(positive_list) < positive_num:
                positive_list.append(point)
        else:
            if len(negative_list) < negative_num:
                negative_list.append(point)

    return positive_list, negative_list


def generate_testing_point(single_formula, category, dimension, number, lower_bound, large_bound):
    number_of_point = int(math.pow(number, (1.0 / dimension)))
    output = []
    if number_of_point == 1:
        for i in range(number):
            point = []
            for j in range(dimension):
                value = np.random.uniform(lower_bound, large_bound)
                point.append(value)
            output.append(point)
    else:
        step = (large_bound - lower_bound) / float(number_of_point)
        x_list = []
        for i in range(number_of_point):
            x_list.append(lower_bound + i * step)
        output = list(product(x_list, repeat=dimension))

    label_list = []
    point_list = []
    for i in output:
        i = list(i)
        formula_tester = lt.FormulaLabelTester(single_formula)
        if category == formula.POLYHEDRON:
            is_true = formula_tester.polycircle_model(single_formula.get_formula()[0],
                                                      single_formula.get_formula()[1], i)
        else:
            is_true = formula_tester.polynomial_model(single_formula.get_formula()[:-1], i,
                                                      single_formula.get_formula()[-1])

        if is_true:
            label_list.append([1.0])
        else:
            label_list.append([0.0])
        point_list.append(i)
    return point_list, label_list

# generate_testing_point(2, 4000, -1.5, 1.5)
# generate_partitioned_random_points_for_polynomial([[1,2],[3],[4,5,6]])
