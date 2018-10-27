import csv
import math
import random
from itertools import product

from prj_test import formula
import label_tester as lt


def generate_data_points(formu, category, lower_bound, upper_bound, data_point_number):
    if category == formula.POLYHEDRON:
        train_path, test_path = generate_random_points_for_sphere(formu, lower_bound, upper_bound, data_point_number)
    elif category == formula.POLYNOMIAL:
        train_path, test_path = generate_random_points_for_polynomial(formu, lower_bound, upper_bound, data_point_number)
    return train_path, test_path


def generate_random_points_for_polynomial(formu, lower_bound, upper_bound, data_point_number):
    formula_tester = lt.FormulaLabelTester(formu)

    train_name = "train" + "_".join(str(x) for x in formu.get_formula()) + ".csv"
    test_name = "test" + "_".join(str(x) for x in formu.get_formula()) + ".csv"

    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    coefficient_list = formu.get_formula()[:-1]
    y = formu.get_formula()[-1]
    with open(train_path, 'w', newline='') as csvfile:

        train = csv.writer(csvfile)

        for k in range(data_point_number):
            xList = []
            variable_num = len(coefficient_list)
            for i in range(variable_num):
                xList.append(random.randint(lower_bound, upper_bound))

            flag = formula_tester.polynomial_model(coefficient_list, xList, y)

            optList = []
            if (flag):
                optList.append(1.0)
                optList += xList
                train.writerow(optList)
            else:
                optList.append(0.0)
                optList += xList
                train.writerow(optList)

    generate_testing_point(formu, variable_num, 1000, lower_bound, upper_bound, test_path, formula.POLYNOMIAL)
    return train_path, test_path


# generate random data points for a circle formula
def generate_random_points_for_sphere(formu, upper_bound, lower_bound, data_point_number):  # [[[12,0],[-12,0]],[4,4]]
    number = random.randint(1, 20)
    formu_list = formu.get_formula()
    dim = len(formu_list[0][0])
    print(dim)

    train_name = "train" + "_".join(str(x) for x in formu_list[1]) + ".csv"
    test_name = "test" + "_".join(str(x) for x in formu_list[1]) + ".csv"

    train_path = "./dataset/" + train_name
    test_path = "./dataset/" + test_name

    with open(train_path, 'w', newline="") as csvfile:
        with open(test_path, 'w', newline="") as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)

            for k in range(data_point_number):
                data_point = []
                generated_point = []
                if k % 3 == 0:
                    center = random.randint(0, len(formu_list[0]) - 1)
                    for i in range(dim):
                        generated_point.append(
                            random.uniform(int(formu_list[0][center][i]) - 300, int(formu_list[0][center][i]) + 300))
                else:
                    for i in range(dim):
                        generated_point.append(random.uniform(-1000, 1000))

                formula_tester = lt.FormulaLabelTester(formu)
                flag = formula_tester.polycircle_model(formu_list[0], formu_list[1], generated_point)

                if (flag):
                    data_point.append(1.0)
                    data_point += generated_point

                    train.writerow(data_point)
                else:
                    data_point.append(0.0)
                    data_point += generated_point

                    train.writerow(data_point)
            # polyhedron_point(formu, dim, 4000, test_path, formula.POLYHEDRON)
            generate_testing_point(formu, dim, 1000, lower_bound, upper_bound, test_path, formula.POLYHEDRON)
    return train_path, test_path


def polyhedron_point(single_formula, dimension, number, path, catagory):
    with open(path, 'w', newline="") as csvfile:
        test = csv.writer(csvfile)
        numberOfPoint = int(round(math.pow(int(number / len(single_formula[0])), (1.0 / dimension))))
        # numberOfPoint = 300
        for j in range(len(single_formula[0])):  # j th center point

            large_bound = single_formula[1][j]
            step = (2 * large_bound) / float(numberOfPoint)
            pointList = []
            for i in range(numberOfPoint):
                pointList.append(-large_bound + i * step)

            output = list(product(pointList, repeat=dimension))

            result = []
            for i in output:
                i = list(i)
                for d in range(len(i)):
                    i[d] += single_formula[0][j][d]
                formula_tester = lt.FormulaLabelTester(single_formula)
                flag = formula_tester.polycircle_model(single_formula[0], single_formula[1], i)
                if (flag):
                    i.insert(0, 0.0)
                else:
                    i.insert(0, 1.0)
                test.writerow(i)


def generate_testing_point(single_formula, dimension, number, lower_bound, large_bound, path, catagory):
    with open(path, 'w', newline="") as csvfile:
        number_of_point = int(round(math.pow(number, (1.0 / dimension))))
        step = (large_bound - lower_bound) / float(number_of_point)
        point_list = []
        for i in range(number_of_point):
            point_list.append(lower_bound + i * step)

        output = list(product(point_list, repeat=dimension))
        test = csv.writer(csvfile)
        for i in output:
            i = list(i)
            formula_tester = lt.FormulaLabelTester(single_formula)
            if catagory == formula.POLYHEDRON:
                is_true = formula_tester.polycircle_model(single_formula.get_formula()[0], single_formula.get_formula()[1], i)
            else:
                is_true = formula_tester.polynomial_model(single_formula.get_formula()[:-1], i, single_formula.get_formula()[-1])

            if is_true:
                i.insert(0, 1.0)
            else:
                i.insert(0, 0.0)
            test.writerow(i)

# generate_testing_point(2, 4000, -1.5, 1.5)
# generate_random_points_for_polynomial([[1,2],[3],[4,5,6]])
