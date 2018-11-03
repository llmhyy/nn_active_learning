import xlwt

import benchmark
import label_tester as lt
import mid_point_active_learning as mal
import util
from prj_test import formula_data_point_generation, formula, formula_generator as fg

category = formula.POLYHEDRON
number = 20

upper_bound = 1000
lower_bound = -1000
learning_rate = 0.01
training_epochs = 1000
data_point_number = 200
parts_num = 5

formulas = fg.generate_formula(category, number)
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
# ws.write(1, 9, "iterations")
ws.write(0, 3, "gal")
ws.write(0, 7, "mal")

model_folder = "models/test-method/test-branch"
model_file = "test-branch"


def write_to_excel(f, ben_train_acc, ben_test_acc, gra_list_train, gra_list_test, mid_list_train, mid_list_test, index):
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
    ws.write(index + 1, 9, str(mid_list_train))
    ws.write(index + 1, 10, str(mid_list_test))

    wb.save("polynomial_result.xls")

    # wb.save("polynomial_result.xls")

    # for i, row in enumerate(result):
    #     for j, col in enumerate(row):
    #         if (i == 1):
    #             if (type(model[0]) != list):
    #                 ws.write(i, j, str(col) + "x^" + str(len(model) - j))
    #             else:
    #                 ws.write(i, j, str(col))
    #         else:
    #             ws.write(i, j, col)

    # print()


index = 0
for f in formula_list:
    util.reset_random_seed()
    print(f.get_formula())
    train_set_x, train_set_y, test_set_x, test_set_y = formula_data_point_generation.generate_partitioned_data(f,
                                                                                                               category,
                                                                                                               lower_bound,
                                                                                                               upper_bound,
                                                                                                               50, 50)
    label_tester = lt.FormulaLabelTester(f)
    point_number_limit = 100
    util.reset_random_seed()
    train_acc_list, test_acc_list, data_point_number_list, appended_point_list = mal.generate_accuracy(
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
    util.reset_random_seed()
    index += 1
    train_acc, test_acc = benchmark.generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
                                                      training_epochs, lower_bound, upper_bound, model_folder,
                                                      model_file)

    write_to_excel(f.get_formula(), train_acc, test_acc, [], [], train_acc_list, test_acc_list, index)
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
