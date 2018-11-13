import xlwt
import time
from main import benchmark, label_tester as lt, mid_point_active_learning as mal, util

import tensorflow as tf
from prj_test import formula_data_point_generation, formula, formula_generator as fg


def write_to_excel(f, ben_train_acc, ben_test_acc, gra_list_train, gra_list_test, mid_list_train, mid_list_test, time,
                   index, file_path):
    # TODO
    ws.write(index + 1, 0, str(f))
    ws.write(index + 1, 1, ben_train_acc)
    ws.write(index + 1, 2, ben_test_acc)

    # ws.write(index+1, 3, gra_list_train[-1])
    # ws.write(index+1, 4, gra_list_test[-1])
    # ws.write(index+1, 5, gra_list_train)
    # ws.write(index+1, 6, gra_list_test)

    ws.write(index + 1, 7, mid_list_train[-1])
    ws.write(index + 1, 8, mid_list_test[-1])
    ws.write(index + 1, 9, max(mid_list_train))
    ws.write(index + 1, 10, max(mid_list_test))
    ws.write(index + 1, 11, str(mid_list_train))
    ws.write(index + 1, 12, str(mid_list_test))
    ws.write(index + 1, 13, time)
    wb.save(file_path)

category = formula.POLYHEDRON
number = 100
upper_bound = 1000
lower_bound = -1000
learning_rate = 0.01
training_epochs = 5000
dimension_range = 2
util.PLOT_MODEL = False

for dimension in range(dimension_range):
    dimension += 1
    if dimension < 2:
        continue

    formulas = fg.generate_formula(category, number, dimension)
    formula_list = formulas.get(category)
    wb = xlwt.Workbook()
    ws = wb.add_sheet(category)
    ws.write(1, 0, "formula")
    ws.write(0, 1, "benchmark")
    ws.write(1, 1, "train")
    ws.write(1, 2, "test")
    ws.write(1, 3, "train")
    ws.write(1, 4, "test")
    # ws.write(1, 5, "iterations")
    ws.write(1, 7, "train")
    ws.write(1, 8, "test")
    ws.write(1, 9, "train best")
    ws.write(1, 10, "test best")
    ws.write(1, 13, "time")
    # ws.write(1, 9, "iterations")
    ws.write(0, 3, "gal")
    ws.write(0, 7, "mal")

    model_folder = "models/test-method/test-branch"
    model_file = "test-branch"

    index = 0
    for f in formula_list:
        start_time = time.time()
        util.reset_random_seed()
        print(f.get_formula())
        train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f,
                                                                                                                   category,
                                                                                                                   lower_bound,
                                                                                                                   upper_bound,
                                                                                                                   50, 50)
        label_tester = lt.FormulaLabelTester(f)
        train_set_x_info = label_tester.check_info(train_set_x)
        point_number_limit = 100
        tf.reset_default_graph()
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
        train_acc_list, test_acc_list, data_point_number_list, appended_point_list = mid_point_learner.generate_accuracy()

        tf.reset_default_graph()
        util.reset_random_seed()
        index += 1
        train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                          training_epochs, lower_bound, upper_bound, model_folder,
                                                          model_file)
        end_time = time.time()
        time_used = end_time - start_time

        file_path = "result/" + "polysphere-" + str(dimension) + ".xls"

        write_to_excel(f.get_formula(), train_acc, test_acc, [], [], train_acc_list, test_acc_list, time_used, index, file_path)
        '''
        ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file,learning_rate, training_epochs, lower_bound, upper_bound)
        #TODO gra_list should contain a set of gra_train_acc and gra_test_acc
        try:
            gra_list = gal.generate_accuracy([],[],train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound, parts_num, True,"", "", True)
        except:
            continue
        #TODO mid_list should contain a set of mid_train_acc and mid_test_acc
        try:
            mid_list = mal.generate_accuracy([],[],train_data_file, test_data_file, f, category, learning_rate, training_epochs, lower_bound, upper_bound, parts_num, True, "", "", True)
        except:
            continue
        index += 1
        print("********************Final result here: ")
        # print(ben_train_acc, ben_test_acc, gra_list, mid_list)
    
        #TODO write to excel once
        write_to_excel(f.get_formula(), ben_train_acc, ben_test_acc, gra_list, mid_list, index)
    
        '''
