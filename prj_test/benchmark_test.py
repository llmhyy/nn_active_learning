import tensorflow as tf

from main import benchmark, label_tester as lt, mid_point_active_learning as mal, util
from prj_test import formula_data_point_generation, formula


def generate_specific_formula():
    formulas = formula.Formulas()
    formu = formula.Formula(
        # [[[-2, 60], [163, -899]], [485, 430]], formula.POLYHEDRON)
        # [[[-700, -700], [700, 700], [-700, 700], [700, -700]], [300, 300, 300, 300]], formula.POLYHEDRON)
        [[[-323, 982], [-798, -621]], [468, 418]], formula.POLYHEDRON)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    # [[[-571, 31]], [445]], formula.POLYHEDRON)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    formulas.put(formu.get_category(), formu)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas


category = formula.POLYHEDRON
formulas = generate_specific_formula()
formula_list = formulas.get(category)
f = formula_list[0]

model_folder = "models/test-method/test-branch"
model_file = "test-branch"

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 1000

# train_data_file = "dataset/train485_430.csv"
# test_data_file = "dataset/test485_430.csv"

util.reset_random_seed()
# train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound,
#                                                                              data_point_number)
# train_set_x, train_set_y, test_set_x, test_set_y = util.read_data_from_file(train_data_file, test_data_file, read_next=True)


train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f, category,
                                                                                                           lower_bound,
                                                                                                           upper_bound,
                                                                                                           50, 50)

path = "dataset/data2.csv"
train_set_x, train_set_y = formula_data_point_generation.read_from_file(path)


tf.reset_default_graph()
util.reset_random_seed()
train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                  training_epochs, lower_bound, upper_bound, model_folder, model_file)

print("benchmark train accuracy", train_acc, "benchmark test accuracy", test_acc)
