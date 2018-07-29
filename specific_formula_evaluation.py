import formula
import formula_generator as fg
import benchmark
import gradient_active_learning as gal
import mid_point_active_learning as mal
import data_point_generation
import xlwt
import random
import numpy as np
import tensorflow as tf


random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


def generate_specific_formula():
    formulas = formula.Formulas()
    # formu = formula.Formula([[[3,5], [-6,2]], [1,4]], formula.POLYHEDRON)
    formu = formula.Formula([[1], [1], 0], formula.POLYNOMIAL)
    formulas.put(formu.get_category(), formu)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas


category = formula.POLYNOMIAL

formulas = generate_specific_formula()
formula_list = formulas.get(category)

f = formula_list[0]

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 100

train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound)
train_data_file = "dataset/train[1]_[1]_0.csv"
test_data_file = "dataset/test[1]_[1]_0.csv"

# train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound)

# ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file, learning_rate, training_epochs)
# TODO gra_list should contain a set of gra_train_acc and gra_test_acc
# gra_list = gal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)
# TODO mid_list should contain a set of mid_train_acc and mid_test_acc
mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)
print("********************Final result here: ")
