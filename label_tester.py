import testing_function as testf
import json_handler
import json
import formula
from sys import stdin


class LabelTester:
    def __init__(self):
        pass

    def test_label(self, points):
        pass


class FormulaLabelTester(LabelTester):
    def __init__(self, test_formula):
        self.test_formula = test_formula

    def test_label(self, points):
        category = self.test_formula.get_category()
        form = self.test_formula.get_formula()
        flagList = []
        if category == formula.POLYHEDRON:
            if isinstance(points[0], list):
                for point in points:
                    flag = testf.polycircle_model(form[0], form[1], point)
                    if (flag):
                        flagList.append(1)
                    else:
                        flagList.append(0)
            else:
                flag = testf.polycircle_model(form[0], form[1], points)
                if (flag):
                    flagList.append(1)
                else:
                    flagList.append(0)
        elif category == formula.POLYNOMIAL:
            if isinstance(points[0], list):
                for point in points:
                    flag = testf.polynomial_model(form[:-1], point, form[-1])
                    if (flag):
                        flagList.append(1)
                    else:
                        flagList.append(0)
            else:
                flag = testf.polynomial_model(form[:-1], points, form[-1])
                if (flag):
                    flagList.append(1)
                else:
                    flagList.append(0)
        return flagList


class CoverageLabelTester(LabelTester):
    def __init__(self, variables):
        self.vars = variables

    def test_label(self, points):
        json_handler.request_label(points, self.variables)
        message_type = stdin.readline()
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        flagList = json_handler.parse_label(data)
        return flagList
