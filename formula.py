POLYNOMIAL = "polynomial"
POLYHEDRON = "polyhedron"


class Formulas():
    def __init__(self):
        self.formulas = {}
        self.formulas[POLYNOMIAL] = []
        self.formulas[POLYHEDRON] = []

    def put(self, category, formula):
        self.formulas[category].append(formula)

    def get(self, category, index):
        return self.formulas[category][index]


