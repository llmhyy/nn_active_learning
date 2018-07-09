import formula
import random

def generate_formula(category, number):
    formulas = formula.Formulas()
    for i in range(number):
        if(category==formula.POLYNOMIAL):
            f = generate_polynomial()
            formulas.put(category, f)
        elif(category==formula.POLYHEDRON):
            f = generate_polyhedron()
            formulas.put(category, f)

    return formulas

def generate_polynomial():
    #TODO variable number is up to 10
    print()

def generate_polyhedron():
    #TODO variable number is up to 10
    # generate center point and radius and number of circles
    num_of_center = random.randint(1, 10)
    num_of_dimension = random.randint(2, 10)
    centers = []
    radiuses = []
    for i in range(num_of_center):
        center = []
        for j in range(num_of_dimension):
            center_coordinate = random.randint(-1000, 1000)
            center.append(center_coordinate)
        radius = random.randint(1,50)
        centers.append(center)
        radiuses.append(radius)
    formula = [centers, radiuses]

    return formula

def generate_specific_formula():
    formulas = formula.Formulas()
    formulas.put(formula.POLYNOMIAL, [1, 2])
    formulas.put(formula.POLYHEDRON, [[[1, 1], [-1, -1]], [0.5, 0.5]])
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas

print (generate_formula("polyhedron", 5).formulas)