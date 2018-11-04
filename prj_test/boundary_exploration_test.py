import benchmark
import label_tester as lt
import util
import boundary_exploration as be
import tensorflow as tf
from prj_test import formula_data_point_generation, formula


def generate_formulas():
    formulas = formula.Formulas()
    formu0 = formula.Formula(
        [[[0, 0]], [500]], formula.POLYHEDRON)
    # [[[-571, 31]], [445]], formula.POLYHEDRON)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    formulas.put(formu0.get_category(), formu0)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    formu1 = formula.Formula(
        [[[500, 0]], [50]], formula.POLYHEDRON)
    formulas.put(formu1.get_category(), formu1)
    return formulas


category = formula.POLYHEDRON
formulas = generate_formulas()
formula_list = formulas.get(category)

parent_formula = formula_list[0]
child_formula = formula_list[1]

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 1000
parts_num = 5

data_point_number = 200
util.reset_random_seed()

train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(
    parent_formula, category,
    lower_bound,
    upper_bound,
    50, 50)

label_tester = lt.FormulaLabelTester(parent_formula)
point_number_limit = 200
util.reset_random_seed()
model_folder = "models/test-method/test-branch"
model_file = "test-branch"

util.reset_random_seed()
train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                  training_epochs, lower_bound, upper_bound, model_folder, model_file)

child_label_tester = lt.FormulaLabelTester(child_formula)
train_set_x1, train_set_y1, _, _ = formula_data_point_generation.generate_partitioned_data(
    child_formula, category,
    -400,
    400,
    0, 50)

be.boundary_explore(train_set_x1, 1, 0, model_folder, model_file, child_label_tester, 10)
