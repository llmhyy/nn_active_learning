import tensorflow as tf
import math
from main import benchmark, label_tester as lt, mid_point_active_learning as mal, util
from prj_test import formula_data_point_generation, formula_generator, formula
# util.reset_random_seed()
number = 1
dimension = 10000
category = formula.POLYHEDRON
formula = formula_generator.generate_formula(category, number, dimension)
formula_list = formula.get(category)
model_folder = "models/test-method/test-branch"
model_file = "test-branch"
f = formula_list[0]

print(f.get_formula())
lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 1000
parts_num = 5
data_point_number = 200
tf.reset_default_graph()
util.reset_random_seed()

train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f, category,
                                                                                                           lower_bound,
                                                                                                           upper_bound,
                                                                                                           200, 200)
# print (train_set_x)
# print (train_set_y)
# label_tester = lt.FormulaLabelTester(f)
# train_set_x_info = label_tester.check_info(train_set_x)
# point_number_limit = 200
# util.reset_random_seed()

# mid_point_learner = mal.MidPointActiveLearner(
#     train_set_x_info,
#     train_set_x[0:50],
#     train_set_y[0:50],
#     test_set_x,
#     test_set_y,
#     learning_rate,
#     training_epochs,
#     lower_bound,
#     upper_bound, False,
#     label_tester,
#     point_number_limit,
#     model_folder,
#     model_file)
# train_acc_list, test_acc_list, data_point_number_list, appended_point_list = mid_point_learner.generate_accuracy()
lay1num=int(math.log2(dimension))
# lay2num=lay1num-1
for i in range(lay1num):
    for j in range(i):

        tf.reset_default_graph()
        util.reset_random_seed()
        hidden1=math.pow(2,i+1)
        hidden2=math.pow(2,j+1)

        train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                  training_epochs, lower_bound, upper_bound, model_folder,
                                                  model_file,hidden1,hidden2)

        print("benchmark train accuracy", train_acc, "benchmark test accuracy", test_acc,"layer number",hidden1,",",hidden2)
# print("midpoint train accuracy", train_acc_list)
# print("midpoint test accuracy", test_acc_list)
# print("midpoint data point number", data_point_number_list)
# for i in range(len(appended_point_list)):
#     appending_dict = appended_point_list[i]
#     print("the", i + 1, "th iteration:")
#     print("  generalization_validation", appending_dict["generalization_validation"])
#     print("  mid_point", appending_dict["mid_point"])

# print("********************Final result here: ")
