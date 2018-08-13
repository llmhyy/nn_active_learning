import random

import formula


def generate_formula(category, number):
    formulas = formula.Formulas()
    for i in range(number):
        if (category == formula.POLYNOMIAL):
            f = generate_polynomial(category)
            formulas.put(category, f)
        elif (category == formula.POLYHEDRON):
            f = generate_polyhedron(category)
            formulas.put(category, f)

    return formulas


def generate_polynomial(category):
    # TODO variable number is up to 10
    y = random.randint(-10, 10)
    variableNumber = random.randint(3, 5)
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


def generate_polyhedron(category):
    # TODO variable number is up to 10
    # generate center point and radius and number of circles
    num_of_center = random.randint(3, 5)
    num_of_dimension = random.randint(3, 5)
    # num_of_center = 1
    # num_of_dimension = 2
    centers = []
    radiuses = []
    for i in range(num_of_center):
        center = []
        for j in range(num_of_dimension):
            center_coordinate = random.randint(-10, 10)
            center.append(center_coordinate)
        radius = random.randint(1, 3)
        centers.append(center)
        radiuses.append(radius)
    formula_ = [centers, radiuses]

    form = formula.Formula(formula_, category)
    return form
