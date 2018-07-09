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
    variableNumber=random.randint(1,10)
    print(variableNumber)
    coefficientList=[]
    for i in range(variableNumber):
        powerNumber=random.randint(1,4)
        print(powerNumber)
        tmpList=[]
        for j in range (powerNumber):
            tmpList.append(random.randint(1,5))
        coefficientList.append(tmpList)
    print (coefficientList)
    return coefficientList

def generate_polyhedron():
    #TODO variable number is up to 10
    print()

def generate_specific_formula():
    formulas = formula.Formulas()
    formulas.put(formula.POLYNOMIAL, [1, 2])
    formulas.put(formula.POLYHEDRON, [[[1, 1], [-1, -1]], [0.5, 0.5]])
    # formulas.put([[[12,0],[-12,0]],[4,4]])

    return formulas

f=generate_formula(formula.POLYNOMIAL,5)
print (f.formulas)