from main import communication
from main import json_handler
import json
from sys import stdin


class InfoChecker:
    def __init__(self):
        pass

    def check_info(self, points):
        raise Exception('InfoChecker class should not initiated')


class FormulaInfoChecker(InfoChecker):
    def __init__(self):
        pass

    def check_info(self, points):
        size = len(points)
        dimension_length = len(points[0])
        total_list = []
        for i in range(size):
            info_list = []
            for j in range(dimension_length):
                dictionary = {'IS_PADDING': False, 'INFLUENTIAL_END': -1, 'MODIFIABLE': True, 'INFLUENTIAL_START': -1,
                              'VALUE': 0,
                              'TYPE': 'DOUBLE', 'NAME': 'a'}
                info_list.append(dictionary)
            total_list.append(info_list)
        return total_list


class CoverageInfoChecker(InfoChecker):
    def __init__(self, variables):
        self.vars = variables

    def check_info(self, points):
        request_string = json_handler.generate_point_info_request(points, self.vars)
        communication.send_point_info_request(request_string)

        message_type = stdin.readline()
        data = stdin.readline()
        data = data.strip("\n")
        data = json.loads(data)
        flag_list = json_handler.parse_point_info(data)

        print("receive data: ", data)

        return flag_list
