import csv
import math
import random
from itertools import product

import formula
import testing_function as tf


def generate_data_points(formu, category, lower_bound, upper_bound):
    if (category == formula.POLYHEDRON):
        trainpath, testpath = random_polyhedron(formu, lower_bound, upper_bound)
    elif (category == formula.POLYNOMIAL):
        trainpath, testpath = generate_random_points_for_polynomial(formu, lower_bound, upper_bound)
    return trainpath, testpath


# TODO coefficient and xList should comes from formu
def generate_random_points_for_polynomial(formu, lower_bound, upper_bound):
    train_name = "train" + "_".join(str(x) for x in formu.get_list()) + ".csv"
    test_name = "test" + "_".join(str(x) for x in formu.get_list()) + ".csv"

    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    coefficient_list = formu.get_list()[:-1]
    y = formu.get_list()[-1]
    with open(train_path, 'w', newline='') as csvfile:

        train = csv.writer(csvfile)

        for k in range(700):
            xList = []
            variable_num = len(coefficient_list)
            for i in range(variable_num):
                xList.append(random.randint(lower_bound, upper_bound))

            flag = tf.polynomial_model(coefficient_list, xList, y)

            optList = []
            if (flag):
                optList.append(1.0)
                optList += xList
                train.writerow(optList)
            else:
                optList.append(0.0)
                optList += xList
                train.writerow(optList)

    testing_point(formu, variable_num, 1000, lower_bound, upper_bound, test_path, formula.POLYNOMIAL)
    return train_path, test_path


# generate random data points for a circle formula
#TODO use upper_bound, lower_bound parameter
def random_polyhedron(formu, upper_bound, lower_bound):  # [[[12,0],[-12,0]],[4,4]]
    number = random.randint(1, 20)
    formu_list = formu.get_list()
    dim = len(formu_list[0][0])
    print(dim)

    train_name = "train" + "_".join(str(x) for x in formu_list[1]) + ".csv"
    test_name = "test" + "_".join(str(x) for x in formu_list[1]) + ".csv"

    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    with open(train_path, 'w') as csvfile:
        with open(test_path, 'w') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)

            for k in range(700):
                data_point = []
                generated_point = []
                if k % 3 == 0:
                    center = random.randint(0, len(formu_list[0]) - 1)
                    for i in range(dim):
                        generated_point.append(
                            random.uniform(int(formu_list[0][center][i]) - 10, int(formu_list[0][center][i]) + 10))
                else:
                    for i in range(dim):
                        generated_point.append(random.uniform(-10, 10))

                flag = tf.polycircle_model(formu_list[0], formu_list[1], generated_point)

                if (flag):
                    data_point.append(0.0)
                    data_point += generated_point

                    train.writerow(data_point)
                else:
                    data_point.append(1.0)
                    data_point += generated_point

                    train.writerow(data_point)
            # polyhedron_point(formu, dim, 4000, test_path, formula.POLYHEDRON)
            testing_point(formu, dim, 1000, lower_bound, upper_bound, test_path, formula.POLYHEDRON)
    return train_path, test_path


def polyhedron_point(formu, dimension, number, path, catagory):
    with open(path, 'w') as csvfile:
        test = csv.writer(csvfile)
        numberOfPoint = int(round(math.pow(int(number / len(formu[0])), (1.0 / dimension))))
        # numberOfPoint = 300
        for j in range(len(formu[0])):  # j th center point

            largebound = formu[1][j]
            step = (2 * largebound) / float(numberOfPoint)
            pointList = []
            for i in range(numberOfPoint):
                pointList.append(-largebound + i * step)

            output = list(product(pointList, repeat=dimension))

            result = []
            for i in output:
                i = list(i)
                for d in range(len(i)):
                    i[d] += formu[0][j][d]
                flag = tf.polycircle_model(formu[0], formu[1], i)
                if (flag):
                    i.insert(0, 0.0)
                else:
                    i.insert(0, 1.0)
                test.writerow(i)


def testing_point(formu, dimension, number, lower_bound, large_bound, path, catagory):
    with open(path, 'w', newline='') as csvfile:
        numberOfPoint = int(round(math.pow(number, (1.0 / dimension))))
        step = (large_bound - lower_bound) / float(numberOfPoint)
        pointList = []
        for i in range(numberOfPoint):
            pointList.append(lower_bound + i * step)

        output = list(product(pointList, repeat=dimension))
        test = csv.writer(csvfile)
        for i in output:
            i = list(i)

            if catagory == formula.POLYHEDRON:
                flag = tf.polycircle_model(formu.get_list()[0], formu.get_list()[1], i)
            else:
                flag = tf.polynomial_model(formu.get_list()[:-1], i, formu.get_list()[-1])

            if (flag):
                i.insert(0, 1.0)
            else:
                i.insert(0, 0.0)
            test.writerow(i)

# testing_point(2, 4000, -1.5, 1.5)
# generate_random_points_for_polynomial([[1,2],[3],[4,5,6]])
