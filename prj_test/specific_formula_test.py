import tensorflow as tf

from main import label_tester as lt, mid_point_active_learning as mal, util, benchmark
from prj_test import formula_data_point_generation, formula


def generate_specific_formula():
    formula_list = formula.Formulas()
    formu = formula.Formula(
        # [[[-2, 60], [163, -899]], [485, 430]], formula.POLYHEDRON)
        # [[[-700, -700], [700, 700], [-700, 700], [700, -700]], [300, 300, 300, 300]], formula.POLYHEDRON)
        [[[-271, 95], [597, -840]], [423, 446]], formula.POLYHEDRON)
        # [[-2, 0, -4, 1], [-5, 3, 4], [5, -2, 5], [-4, 5, -5], [1, -3, -3, 1], [1, -4], [-5, 5, 0, -2], [2, 4, -3, -5], [-4, 5, 5, -2], [1, 3, -3], [-1, 0], [-5], [-2, 2, 3], [3], [-5, 1], [-5, -3, -3, -2], [-1], [4, 3, -1, -4], [-2, 0, 2, -4], [1, 4, 0, 4], [-1, 5], [-1], [-1], [-4, -2, -4], [-1], [2], [4, 1, 1], [1, -4, 3, 0], [5, 0], [-3], [-5], [1], [4, 5], [-3], [-1, 0], [0, 1], [3, 4], [-2, 0, 2], [5, 5], [2, 0, -3], [1], [3, -1, 2, -4], [-1, 3, -2], [1, 3, 3], [-2, -2, -3, 2], [-3], [0, -5, -5], [3, -4], [-4, -3, 5, -1], [1, 2, 5], [-1, 0], [-1, 0, 4], [4, 5, -5], [3, 0, 0, -4], [-4, 4, 5, 5], [-1, -3, 5, 4], [-1, 4, 4, 0], [-2, -5, 5, -1], [1], [-1, -3], [3, 5, 3, -4], [0, -4, 3], [4], [-1, -3, -4], [-1, 1, -1, 4], [-3], [3, 0], [2], [-2, -4], [-2, 5, 5], [4, 4, 5], [-1, 0], [4, 3], [3, 1], [-1, 4, -4], [1, 4, -1], [-2, 4, -3], [4, 5, 2], [1, 2, -1, -5], [-5, 3, -5], [1, -4], [-3], [-3, 2], [1, -3, 0, 4], [5, 1], [0, -4, 0, 5], [-2, 0], [3, -5, 3, -2], [4, 4, 2], [-3, 4], [3], [0, 5], [-2], [2, 1, -5], [-1, 3, -1], [-4, 1], [-4, -3, 4], [3, 3, -3, 2], [-4], [-3, -1, 2], 8],formula.POLYNOMIAL)
        # [[5, -3, 5], [3], [-2], [-3, 4, -2], [3, -4], [-3, -5, -4], [-5], [1, 3], [-4, 3, -2], [1, 4, 1, 4], [-1, 0, -4], [-4, -5, -5], [4, -4, 1], [2, 1, 4, 4], [1, -4, 2, -1], [2, 1, 2], [4], [-3], [-5, 2, -3], [-5, 0], [0, -4], [4, 2], [0, -1, 3, 0], [3, 1], [4, -1, -4, 5], [-2, 5, 1], [0, -5], [2, 2, -4, 1], [-4], [-4, -4, 4, 0], [-3, -1, 3], [-2, 3], [-4, 1], [4], [5], [-3], [-1], [0, -4], [4, 0, 5], [-4, -5, -5, 5], [-2, 5], [-2, 4], [-3, 4], [5, 3, -1, 5], [4], [5, 0], [0, -3, 2, -5], [0, 5, 2, 1], [0, 3], [0, 5, 2], [3, -1, 2], [-2, 4, -3, 5], [-3, 1, -2], [-1, 5, -1, -1], [-3, 4, -2], [-1, -3], [-3, -2], [-2, -2], [4, -4, -3, -3], [5, -3, -3], [4, 4, 5, -1], [2, 4, -4], [1], [2], [4, 3], [5, 1], [-2, -3, -2], [0, 0, 5], [3, 5, 0, 2], [-4, -1, 0], [-4, -5], [-4, 0, 5], [-2, -4, -5, 3], [2, -3, -3], [1, -5, -2], [3], [-2], [5, -4, 0, 3], [3], [1, -3, -2], [4], [-3, 0, 0], [3, -4], [2, -3], [0, 2, 2], [-5, 3], [0, 1], [4, -2, 2], [-5, 1], [4, 1], [2, 4], [5], [-3, -4, -5, 3], [3, -3, 2], [2], [2, 2, -4, 4], [1], [1, -1], [4], [1, 3], 3],formula.POLYNOMIAL)
    # [[3, -5], [1, -3, 2], [-1], [1, 0], [5, -4, 3, 4], [5, -3, 1], [3], [5, -1, 1, -2], [-2, 2, -1, -5], [2, -2], [0], [3, -3], [-3, -1], [-2, 3], [-1], [-2], [-5, 1, -3], [0, 3], [-5, 5], [-4, -5, 3, 5], [-4], [2, 0, -1, -3], [4, -2], [4, 2, 0, -2], [1], [-1, 4, -1, 2], [-2, -2], [2], [-3], [2, -4, 5], [-3], [-1, -1, -3, 0], [-4, -4], [-3], [5, 2, 4], [3, 5, 1, -3], [-3, -5], [5, -5, 0], [-5, 2, 4], [-3, -3], [1], [5], [2, -3, 0, -2], [-2], [4, -1, 5, -5], [2, 2, 1], [4, 4], [0, 3, -5], [-4, 0, -3, -3], [-1], [5, -1, -1, -2], [5, 2, -5, 4], [-5, -3, -5, 4], [5, -2, 3, 4], [-5], [1, -4, 3, 5], [3, -3, -3], [-5, 1, -3], [-2, -1, 4], [0, 3], [4], [-2, -2, -1, -5], [-1, -3, -4], [-2, -3, 4, 3], [0], [-1], [-1], [-3, 1], [0, -3, -3, -2], [3, -3, 0, -3], [5, -5], [-4, -5, 2], [-1], [-4, 0, 1, -5], [2, 2, -5], [1, 5, 0, 3], [0, 2], [-4, 0, 0, -2], [2], [1, -2], [-1, -2, -3, 5], [1], [3], [-3, -2, 4, -5], [-2], [2], [3, 1], [3, 0, 5, 3], [5, -3, -1], [4, 3, 5], [2, 2, 0], [-4, -5, -1, 5], [0, 4, -1], [-1], [4, -3, -4], [2, 1, -5, 5], [4, 2, 0, 4], [1], [1, 0, -2], [2, -2, 5, -4], -8],formula.POLYNOMIAL)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    # [[[-571, 31]], [445]], formula.POLYHEDRON)
    # [[[0, 0]], [500]], formula.POLYHEDRON)
    formula_list.put(formu.get_category(), formu)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formula_list


category = formula.POLYHEDRON
formulas = generate_specific_formula()
formula_list = formulas.get(category)
f = formula_list[0]

model_folder = "models/test-method/test-branch"
model_file = "test-branch"

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 5000

# train_data_file = "dataset/train485_430.csv"
# test_data_file = "dataset/test485_430.csv"
tf.reset_default_graph()
util.reset_random_seed()


# train_data_file, test_data_file = data_point_generation.generate_data_points(f, category, lower_bound, upper_bound,
#                                                                              data_point_number)
# train_set_x, train_set_y, test_set_x, test_set_y = util.read_data_from_file(train_data_file, test_data_file, read_next=True)


train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f, category,
                                                                                                           lower_bound,
                                                                                                           upper_bound,
                                                                                                           50, 50)

label_tester = lt.FormulaLabelTester(f)
train_set_x_info = label_tester.check_info(train_set_x)
point_number_limit = 100

util.reset_random_seed()
mid_point_learner = mal.MidPointActiveLearner(
    train_set_x_info,
    train_set_x[0:50],
    train_set_y[0:50],
    test_set_x,
    test_set_y,
    learning_rate,
    training_epochs,
    lower_bound,
    upper_bound, False,
    label_tester,
    point_number_limit,
    model_folder,
    model_file)
train_acc_list, test_acc_list, data_point_number_list, appended_point_list = mid_point_learner.train()

tf.reset_default_graph()
util.reset_random_seed()
train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                  training_epochs, lower_bound, upper_bound, model_folder, model_file)

print("benchmark train accuracy", train_acc, "benchmark test accuracy", test_acc)
print("midpoint train accuracy", train_acc_list)
print("midpoint test accuracy", test_acc_list)
print("midpoint data point number", data_point_number_list)
for i in range(len(appended_point_list)):
    appending_dict = appended_point_list[i]
    print("the", i + 1, "th iteration:")
    print("  generalization_validation", appending_dict["generalization_validation"])
    print("  mid_point", appending_dict["mid_point"])
