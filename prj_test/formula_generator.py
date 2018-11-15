import random

from prj_test import formula


def generate_formula(category, number, dimension):
    formulas = formula.Formulas()
    for i in range(number):
        if (category == formula.POLYNOMIAL):
            f = generate_polynomial(category, dimension)
            formulas.put(category, f)
        elif (category == formula.POLYHEDRON):
            f = generate_polyhedron(category, dimension)
            formulas.put(category, f)

    return formulas


def generate_polynomial(category, dimension):
    # TODO variable number is up to 10
    y = random.randint(-10, 10)
    variableNumber = dimension
    coefficientList = []
    for i in range(variableNumber):
        powerNumber = random.randint(1, 4)

        tmpList = []
        for j in range(powerNumber):
            tmpList.append(random.randint(-5, 5))
        coefficientList.append(tmpList)
    coefficientList.append(y)

    form = formula.Formula(coefficientList, category)
    return form


def generate_polyhedron(category, num_of_dimension):
    # generate center point and radius and number of circles
    num_of_center = random.randint(1, 5)
    centers = []
    radius_list = []
    for i in range(num_of_center):
        center = []
        for j in range(num_of_dimension):
            center_coordinate = random.randint(-1000, 1000)
            center.append(center_coordinate)
        radius = random.randint(100, 500)
        centers.append(center)
        radius_list.append(radius)
    formula_ = [centers, radius_list]

    form = formula.Formula(formula_, category)
    return form
