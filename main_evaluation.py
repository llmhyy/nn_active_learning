import formula
import formula_generator as fg
import benchmark
##import gradient_active_learning as gal
import mid_point_active_learning as mal
import data_point_generation

category = formula.POLYHEDRON
number = 5

formulas = fg.generate_formula(category, number)
formula_list = formulas.get(category)

def write_to_excel(f, ben_train_acc, ben_test_acc, gra_list, mid_list):
    #TODO
    print()

for f in formula_list:

    # f = [[-1,4,2,5],[-2,5,1,0],-1748]
    #TODO each foumla write its generated data into files with the formula name
    train_data_file, test_data_file = data_point_generation.generate_data_points(f, category)

    #ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file)
    #TODO gra_list should contain a set of gra_train_acc and gra_test_acc
    #gra_list = gal.generate_accuracy(train_data_file, test_data_file, f, category)
    #TODO mid_list should contain a set of mid_train_acc and mid_test_acc
    mid_list = mal.generate_accuracy(train_data_file, test_data_file,f,category)
    break
    #TODO write to excel once
    #write_to_excel(f, ben_train_acc, ben_test_acc, gra_list, mid_list)

