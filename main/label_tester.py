import json
import math
from sys import stdin

from main import communication, json_handler
from prj_test import formula


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
                    flag = self.polycircle_model(form[0], form[1], point)
                    if (flag):
                        flag_list.append(1)
                    else:
                        flag_list.append(0)
            else:
                flag = self.polycircle_model(form[0], form[1], points)
                if (flag):
                    flag_list.append(1)
                else:
                    flag_list.append(0)
        elif category == formula.POLYNOMIAL:
            if isinstance(points[0], list):
                for point in points:
                    flag = self.polynomial_model(form[:-1], point, form[-1])
                    if (flag):
                        flag_list.append(1)
                    else:
                        flag_list.append(0)
            else:
                flag = self.polynomial_model(form[:-1], points, form[-1])
                if flag:
                    flag_list.append(1)
                else:
                    flag_list.append(0)
        return flag_list

    def polynomial_model(self, coefficient_list, x, y):
        variable_num = len(coefficient_list)
        output = 0
        for i in range(variable_num):
            tmpList = coefficient_list[i]
            tmpLength = len(tmpList)
            for j in range(tmpLength):
                power = tmpLength - j
                output += tmpList[j] * math.pow(x[i], power)

        if (output > y):
            return True
        else:
            return False

    def polycircle_model(self, center, radius, x):
        # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
        for i in range(len(center)):
            point_radius = 0
            for j in range(len(x)):
                point_radius += (x[j] - center[i][j]) * (x[j] - center[i][j])
            if (point_radius < radius[i] * radius[i]):
                return True
        return False


class CoverageLabelTester(LabelTester):
    def __init__(self, variables):
        self.vars = variables

    def test_label(self, points):
        request_string = json_handler.generate_label_request(points, self.vars)
        communication.send_label_request(request_string)

        message_type = stdin.readline()
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        flagList = json_handler.parse_label(data)
        return flagList
