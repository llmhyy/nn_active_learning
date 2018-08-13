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
    formu = formula.Formula([[1, 2, -5, 4], [-4, 0, -3, -5], [0, -3, -2, -4], [-1, -2, 4], [5], 7], formula.POLYNOMIAL)
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
# train_data_file = "dataset/train[5, 2, -1, 3]_[-3, 5, 2, -1]_-10.csv"
# test_data_file = "dataset/test[5, 2, -1, 3]_[-3, 5, 2, -1]_-10.csv"
print (f.get_list())
tf.reset_default_graph()
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file, learning_rate, training_epochs,lower_bound, upper_bound)

tf.reset_default_graph()
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

gra_list = gal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)

tf.reset_default_graph()
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)
# tf.reset_default_graph()
# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.set_random_seed(random_seed)
# mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)

print("********************Final result here: ")
