import json
import math
from sys import stdin

import formula
import json_handler


def logModel(x1, x2):
    if (x2 > math.log(x1)):
        return True
    else:
        return False


def circleModel(x1, x2):
    if ((x1 - 12.5) * (x1 - 12.5) + x2 * x2 < 100 or (x1 + 12.5) * (x1 + 12.5) + x2 * x2 < 100):
        return True
    else:
        return False


def polynomial_model(coefficientList, x, y):
    variableNum = len(coefficientList)
    output = 0
    for i in range(variableNum):
        tmpList = coefficientList[i]
        tmpLength = len(tmpList)
        for j in range(tmpLength):
            power = tmpLength - j

            output += tmpList[j] * math.pow(x[i], power)
    if (output > y):
        return True
    else:
        return False


def polycircle_model(center, radius, x):
    # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
    for i in range(len(center)):
        point_radius = 0
        for j in range(len(x)):
            point_radius += (x[j] - center[i][j]) * (x[j] - center[i][j])
        if (point_radius < radius[i] * radius[i]):
            return True
    return False


def test_label(points, formu, type, name_list, mock):
    if mock == True:
        category = formu.get_category()
        form = formu.get_list()
        flagList = []
        if category == formula.POLYHEDRON:
            if isinstance(points[0], list):
                for point in points:
                    flag = polycircle_model(form[0], form[1], point)
                    if (flag):
                        flagList.append(1)
                    else:
                        flagList.append(0)
            else:
                flag = polycircle_model(form[0], form[1], points)
                if (flag):
                    flagList.append(1)
                else:
                    flagList.append(0)
        elif category == formula.POLYNOMIAL:
            if isinstance(points[0], list):
                for point in points:
                    flag = polynomial_model(form[:-1], point, form[-1])
                    if (flag):
                        flagList.append(1)
                    else:
                        flagList.append(0)
            else:
                flag = polynomial_model(form[:-1], points, form[-1])
                if (flag):
                    flagList.append(1)
                else:
                    flagList.append(0)
        return flagList
    else:
        json_handler.requestLabel(points, type, name_list)
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        flagList = json_handler. \
            label_parser(data)
        return flagList
