import formula
import formula_generator as fg
import benchmark
import gradient_active_learning as gal
import mid_point_active_learning as mal
import data_point_generation
import xlwt


def generate_specific_formula():
    formulas = formula.Formulas()
    formu = formula.Formula([[5], [-1], -6], formula.POLYNOMIAL)
    formulas.put(formu.get_category(), formu)
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas


category = formula.POLYNOMIAL

formulas = generate_specific_formula()
formula_list = formulas.get(category)

f = formula_list[0]

# train_data_file, test_data_file = data_point_generation.generate_data_points(f, category)
train_data_file = "dataset/train[5]_[-1]_-6.csv"
test_data_file = "dataset/test[5]_[-1]_-6.csv"

# ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file)
# ben_train_acc, ben_test_acc = benchmark.generate_accuracy(train_data_file, test_data_file)
# TODO gra_list should contain a set of gra_train_acc and gra_test_acc
# gra_list = gal.generate_accuracy(train_data_file, test_data_file, f, category)
# TODO mid_list should contain a set of mid_train_acc and mid_test_acc
#ben_train_acc, ben_test_acc =benchmark.generate_accuracy(train_data_file,test_data_file)
mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category)
mid_list = mal.generate_accuracy(train_data_file, test_data_file, f, category)
#ben_train_acc, ben_test_acc =benchmark.generate_accuracy(train_data_file,test_data_file)
print("********************Final result here: ")
