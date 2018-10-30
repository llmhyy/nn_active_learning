import label_tester as lt
import mid_point_active_learning as mal
import util
import random
from prj_test import formula_data_point_generation, formula_generator,formula

number=1
category=formula.POLYHEDRON
formula=formula_generator.generate_formula(category,number)
formula_list = formula.get(category)

f=formula_list[0]

print (f.get_formula())
lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 300
parts_num = 5
data_point_number = 100
util.reset_random_seed()
train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f, category,
                                                                                                           lower_bound,
                                                                                                           upper_bound,
                                                                                                           50, 50)

label_tester = lt.FormulaLabelTester(f)
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
for appending_dict in appended_point_list:
    print("generalization_validation", appending_dict["generalization_validation"])
    print("mid_point", appending_dict["mid_point"])