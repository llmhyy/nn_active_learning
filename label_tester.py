import testing_function as testf
import json_handler
import json
import formula
import communication
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
        flag_list = []
        if category == formula.POLYHEDRON:
            if isinstance(points[0], list):
                for point in points:
                    flag = testf.polycircle_model(form[0], form[1], point)
                    if (flag):
                        flag_list.append(1)
                    else:
                        flag_list.append(0)
            else:
                flag = testf.polycircle_model(form[0], form[1], points)
                if (flag):
                    flag_list.append(1)
                else:
                    flag_list.append(0)
        elif category == formula.POLYNOMIAL:
            if isinstance(points[0], list):
                for point in points:
                    flag = testf.polynomial_model(form[:-1], point, form[-1])
                    if (flag):
                        flag_list.append(1)
                    else:
                        flag_list.append(0)
            else:
                flag = testf.polynomial_model(form[:-1], points, form[-1])
                if flag:
                    flag_list.append(1)
                else:
                    flag_list.append(0)
        return flag_list


class CoverageLabelTester(LabelTester):
    def __init__(self, variables):
        self.vars = variables

    def test_label(self, points):
        request_string = json_handler.generate_label_request(points, self.variables)
        communication.send_label_request(request_string)

        message_type = stdin.readline()
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        flagList = json_handler.parse_label(data)
        return flagList
