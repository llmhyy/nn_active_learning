POLYNOMIAL = "polynomial"
POLYHEDRON = "polyhedron"


class Formula():
    def __init__(self, list, category):
        self.list = list
        self.category = category

    def get_formula(self):
        return self.list

    def get_category(self):
        return self.category

    def set_list(self, list):
        self.list = list

    def set_category(self, category):
        self.category = category


class Formulas():
    def __init__(self):
        self.formulas = {}
        self.formulas[POLYNOMIAL] = []
        self.formulas[POLYHEDRON] = []

    def put(self, category, formula):
        self.formulas[category].append(formula)

    def get(self, category):
        return self.formulas[category]


