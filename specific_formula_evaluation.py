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

random_seed = 10
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)


def generate_specific_formula():
    formulas = formula.Formulas()
    formu = formula.Formula(
        # [[[-2, 60], [163, -899]], [485, 430]], formula.POLYHEDRON)
        [[[-700, -700], [700, 700]], [300, 300]], formula.POLYHEDRON)
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
training_epochs = 100
parts_num = 5

train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound)
# train_data_file = "dataset/train485_430.csv"
# test_data_file = "dataset/test485_430.csv"
print(f.get_list())
tf.reset_default_graph()
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

X = [[-10, -10], [-8, -5], [-11, -6], [0, 10], [1, 14], [-1, 9], [89, 55], [68, 86]]
# while True:
#     still_on_one_side, X = cluster.get_clustering_points(X, True, f)
#     if not still_on_one_side:
#         print("different label")
#         break
#     else:
#         print("same label")

# tf.reset_default_graph()
# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.set_random_seed(random_seed)

# gra_list = gal.generate_accuracy([], [], train_data_file, test_data_file, f, category, learning_rate, training_epochs,
#                                  lower_bound, upper_bound, parts_num, False, "", "", True)

tf.reset_default_graph()
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)
train_set_X = []
train_set_Y = []
test_set_X = []
test_set_Y = []
mid_list = mal.generate_accuracy([], [], train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound, parts_num, True, "", "", True)
# tf.reset_default_graph()
# random.seed(random_seed)
# np.random.seed(random_seed)
# tf.set_random_seed(random_seed)
# mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound)

print("********************Final result here: ")
