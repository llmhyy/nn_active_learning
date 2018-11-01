import label_tester as lt
import mid_point_active_learning as mal
import util
import random
import benchmark
from prj_test import formula_data_point_generation, formula


def generate_specific_formula():
    formulas = formula.Formulas()
    formu = formula.Formula(
        # [[[-2, 60], [163, -899]], [485, 430]], formula.POLYHEDRON)
        # [[[-700, -700], [700, 700], [-700, 700], [700, -700]], [300, 300, 300, 300]], formula.POLYHEDRON)
        # [[[917, 617], [404, 193], [-361, 946]], [457, 441, 469]],formula.POLYHEDRON)
        # [[[156, -748], [243, 827], [586, -712]], [493, 456, 450]],formula.POLYHEDRON)
    [[[-228, -586], [785, 782], [-495, -677]], [422, 428, 404]],formula.POLYHEDRON)
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

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 1000
parts_num = 5

# train_data_file = "dataset/train485_430.csv"
# test_data_file = "dataset/test485_430.csv"

data_point_number = 200
util.reset_random_seed()
# train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound,
#                                                                              data_point_number)
# train_set_x, train_set_y, test_set_x, test_set_y = util.read_data_from_file(train_data_file, test_data_file, read_next=True)

train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f, category,
                                                                                                           lower_bound,
                                                                                                           upper_bound,
                                                                                                           50, 50)

label_tester = lt.FormulaLabelTester(f)
point_number_limit = 200
util.reset_random_seed()
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
util.reset_random_seed()
train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                  training_epochs, lower_bound, upper_bound)

print("benchmark train accuracy", train_acc, "benchmark test accuracy", test_acc)
print("midpoint train accuracy", train_acc_list)
print("midpoint test accuracy", test_acc_list)
print("midpoint data point number", data_point_number_list)
for i in range(len(appended_point_list)):
    appending_dict = appended_point_list[i]
    print("the", i+1, "th iteration:")
    print("  generalization_validation", appending_dict["generalization_validation"])
    print("  mid_point", appending_dict["mid_point"])

print("********************Final result here: ")
