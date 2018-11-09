import json
import math
from sys import stdin

from main import communication, json_handler, domain_names as dn
from prj_test import formula


class LabelTester:
    def __init__(self):
        raise Exception("LabelTester should not be instantiated.")

    def test_label(self, points):
        pass

    def check_info(self, points):
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

    def check_info(self, points):
        size = len(points)
        dimension_length = len(points[0])
        total_list = []
        for i in range(size):
            info_list = []
            for j in range(dimension_length):
                dictionary = {dn.IS_PADDING: False, dn.INFLUENTIAL_END: -1, dn.MODIFIABLE: True,
                              dn.INFLUENTIAL_START: -1,
                              dn.VALUE: 0,
                              dn.TYPE: 'DOUBLE', dn.NAME: 'a'}
                info_list.append(dictionary)
            total_list.append(info_list)
        return total_list

    def polynomial_model(self, coefficient_list, x, y):
        variable_num = len(coefficient_list)
        output = 0
        for i in range(variable_num):
            tmp_list = coefficient_list[i]
            tmp_length = len(tmp_list)
            for j in range(tmp_length):
                power = tmp_length - j
                output += tmp_list[j] * math.pow(x[i], power)

        if output > y:
            return True
        else:
            return False

    def polycircle_model(self, center, radius, x):
        # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
        for i in range(len(center)):
            point_radius = 0
            for j in range(len(x)):
                point_radius += (x[j] - center[i][j]) * (x[j] - center[i][j])
            if point_radius < radius[i] * radius[i]:
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
        labels, info = json_handler.parse_label(data)

        print(labels)
        print("receive data: ", data)

        return labels

    def check_info(self, points):
        request_string = json_handler.generate_point_info_request(points, self.vars)
        communication.send_point_info_request(request_string)

        message_type = stdin.readline()
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        info_list = json_handler.parse_point_info(data)

        print(info_list)
        print("receive data: ", data)

        return info_list
