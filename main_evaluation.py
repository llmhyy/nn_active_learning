import formula
import formula_generator as fg
import benchmark
import gradient_active_learning as gal
import mid_point_active_learning as mal
import data_point_generation
import xlwt

category = formula.POLYNOMIAL
number = 50

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
ws.write(1, 5, "iterations")    
ws.write(1, 7, "train")
ws.write(1, 8, "test")
ws.write(1, 9, "iterations")
ws.write(0, 3, "gal")
ws.write(0, 7, "mal")


def write_to_excel(f, ben_train_acc, ben_test_acc, gra_list, mid_list, index):
    #TODO


    ws.write(index+1, 0, str(f))
    ws.write(index+1, 1, str(ben_train_acc))
    ws.write(index+1, 2, str(ben_test_acc))

    ws.write(index+1, 3, gra_list[0][-1])
    ws.write(index+1, 4, gra_list[1][-1])
    ws.write(index+1, 5, len(gra_list[0]))
    ws.write(index+1, 6, str(gra_list))

    ws.write(index+1, 7, mid_list[0][-1])
    ws.write(index+1, 8, mid_list[1][-1])
    ws.write(index+1, 9, len(mid_list[0]))
    ws.write(index+1, 10, str(mid_list))

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
    print(f)
    # f = [[-1,4,2,5],[ -2,5,1,0],-1748]
    #TODO each foumla write its generated data into files with the formula name
    train_data_file, test_data_file = data_point_generation.generate_data_points(f, category)

    ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file)
    #TODO gra_list should contain a set of gra_train_acc and gra_test_acc
    gra_list = gal.generate_accuracy(train_data_file, test_data_file, f, category)
    #TODO mid_list should contain a set of mid_train_acc and mid_test_acc
    try:
        mid_list = mal.generate_accuracy(train_data_file, test_data_file,f,category)
    except:
        continue

    index += 1
    print("********************Final result here: ")
    # print(ben_train_acc, ben_test_acc, gra_list, mid_list)

    #TODO write to excel once
    write_to_excel(f, ben_train_acc, ben_test_acc, gra_list, mid_list, index)
    # break

