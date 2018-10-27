import formula
import formula_generator as fg
import benchmark
import gradient_active_learning as gal
import mid_point_active_learning as mal
import cluster
import data_point_generation
import xlwt
import random
import numpy as np
import tensorflow as tf
import util
import label_tester as lt


def generate_specific_formula():
    formulas = formula.Formulas()
    formu = formula.Formula(
        # [[[-2, 60], [163, -899]], [485, 430]], formula.POLYHEDRON)
        # [[[-700, -700], [700, 700]], [300, 300]], formula.POLYHEDRON)
        [[[0, 0, 0, 0, 0]], [500]], formula.POLYHEDRON)
    # [[[-571, 31]], [445]], formula.POLYHEDRON)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    formulas.put(formu.get_category(), formu)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas


category = formula.POLYHEDRON
formulas = generate_specific_formula()
formula_list = formulas.get(category)

f = formula_list[0]

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 300
parts_num = 5

# train_data_file = "dataset/train485_430.csv"
# test_data_file = "dataset/test485_430.csv"

data_point_number = 100
util.reset_random_seed()
train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound,
                                                                             data_point_number)
train_set_x, train_set_y, test_set_x, test_set_y = util.preprocess(train_data_file, test_data_file, read_next=True)
label_tester = lt.FormulaLabelTester(f)

# util.reset_random_seed()
# train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
#                                                   training_epochs, lower_bound, upper_bound)

point_number_limit = 100
train_acc_list, test_acc_list, data_point_number_list, appended_point_list = mal.generate_accuracy(train_set_x[0:50],
                                                                                                   train_set_y[0:50],
                                                                                                   test_set_x,
                                                                                                   test_set_y,
                                                                                                   learning_rate,
                                                                                                   training_epochs,
                                                                                                   lower_bound,
                                                                                                   upper_bound, False,
                                                                                                   label_tester,
                                                                                                   point_number_limit)

# print("benchmark train accuracy", train_acc, "benchmark test accuracy", test_acc)
print("midpoint train accuracy", train_acc_list)
print("midpoint test accuracy", test_acc_list)
print("midpoint data point number", data_point_number_list)
for appending_dict in appended_point_list:
    print("generalization_validation", appending_dict["generalization_validation"])
    print("mid_point", appending_dict["mid_point"])

# mid_list = mal.generate_accuracy([], [], train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound, parts_num, True, "", "", True)
# tf.reset_default_graph()
# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.set_random_seed(random_seed)
# mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)

print("********************Final result here: ")
