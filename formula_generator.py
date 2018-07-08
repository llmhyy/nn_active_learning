import formula

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
    print()

def generate_specific_formula():
    formulas = formula.Formulas()
    formulas.put(formula.POLYNOMIAL, [1, 2])
    formulas.put(formula.POLYHEDRON, [[[1, 1], [-1, -1]], [0.5, 0.5]])
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas
