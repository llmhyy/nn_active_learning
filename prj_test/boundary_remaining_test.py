from main import benchmark, boundary_remaining as br, label_tester as lt, util
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
        [[[500, 0]], [100]], formula.POLYHEDRON)
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

# util.reset_random_seed()
# train_acc, test_acc = benchmark.train(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate,
#                                                   training_epochs, lower_bound, upper_bound, model_folder, model_file)


positive_x = []
for i in range(len(train_set_x)):
    point = train_set_x[i]
    if train_set_y[i][0] == 1:
        positive_x.append(point)
positive_x_info = label_tester.check_info(positive_x)

boundary_remainer = br.BoundaryRemainer(positive_x_info, positive_x, model_folder, model_file, 10)
new_point_list = boundary_remainer.search_remaining_boundary_points()

labels = label_tester.test_label(new_point_list)
count = 0
for i in range(len(labels)):
    if labels[i] == 1:
        count += 1

print("accuracy", count/len(labels))
