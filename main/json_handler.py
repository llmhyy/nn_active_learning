import json
import os
from main import domain_names as dn
from main import util
from sys import stdout

from main import variable as v


# input = {
#          dn.BRANCH_ID: "2-3",
#           dn.METHOD_ID: "com.method.XX",}
#
# test_data = [[1, 2], [3, 4], [5, 6]]
def parse_model_check(message_body):
    branch_id = message_body[dn.BRANCH_ID]
    method_id = message_body[dn.METHOD_ID]
    return branch_id, method_id


def generate_model_check_response(message_content):
    return {dn.EXISTENCE: message_content}


# input = {
#          dn.METHOD_ID: "com.test.Class.method()",
#          dn.BRANCH_ID: "2-3",
#          "DATA": [
#              [{dn.VALUE: "877", dn.TYPE: dn.INTEGER, dn.NAME: "a"}, {dn.VALUE: "0", dn.TYPE: "PRIMITIVE", dn.NAME: "b"}],
#              [{dn.VALUE: "548", dn.TYPE: "PRIMITIVE", dn.NAME: "a"}, {dn.VALUE: "969", dn.TYPE: "PRIMITIVE", dn.NAME: "b"}]
#          ]}
#
# test_data = [[1, 2], [3, 4], [5, 6]]
def parse_boundary_exploration(message):
    model_folder = message[dn.METHOD_ID]
    model_file_name = message[dn.BRANCH_ID]

    model_folder = os.path.join(model_folder, model_file_name)

    data = message[dn.TEST_DATA]

    sample_point = data[0]
    variables = []
    for dimension in sample_point:
        variable = v.Variable(dimension[dn.NAME], dimension[dn.TYPE])
        variables.append(variable)

    data_set = []
    data_set_info = []
    for d in data:
        point = []
        for dimension in d:
            value = util.parse_value_with_data_type(dimension[dn.VALUE], dimension[dn.TYPE])
            point.append(value)

        data_set.append(point)
        data_set_info.append(d)

    return data_set_info, data_set, model_folder, model_file_name, variables


# input = {
#           dn.METHOD_ID: "com.test.Class.method()",
#           dn.BRANCH_ID: "2-3",
#           dn.POINT_NUMBER_LIMIT: 100
#          dn.POSITIVE_DATA: [
#              {'IS_PADDING': False, 'INFLUENTIAL_END': 2, 'MODIFIABLE': True, 'INFLUENTIAL_START': 1, 'VALUE': 'false', 'TYPE': 'BOOLEAN', 'NAME': 'a.isNull'},
#
#          dn.NEGATIVE_DATA: [
#              {'IS_PADDING': False, 'INFLUENTIAL_END': 2, 'MODIFIABLE': True, 'INFLUENTIAL_START': 1, 'VALUE': 'false', 'TYPE': 'BOOLEAN', 'NAME': 'a.isNull'},
#              {'IS_PADDING': False, 'INFLUENTIAL_END': 2, 'MODIFIABLE': True, 'INFLUENTIAL_START': 1, 'VALUE': 'false', 'TYPE': 'BOOLEAN', 'NAME': 'a.isNull'}
#          ]}
#
# test_data = [[1, 2], [3, 4], [5, 6]]
def parse_training_message_body(message):
    model_folder = message[dn.METHOD_ID]
    model_file_name = message[dn.BRANCH_ID]
    model_folder = os.path.join(model_folder, model_file_name)

    point_number_limit = message[dn.POINT_NUMBER_LIMIT]

    train_set_x_info = []
    train_set_x = []
    train_set_y = []

    positive_data = message[dn.POSITIVE_DATA]
    negative_data = message[dn.NEGATIVE_DATA]

    sample_point = positive_data[0]
    variables = []
    for dimension in sample_point:
        variable = v.Variable(dimension[dn.NAME], dimension[dn.TYPE])
        variables.append(variable)

    for positive_point in positive_data:
        point = []
        for dim in positive_point:
            value = util.parse_value_with_data_type(dim[dn.VALUE], dim[dn.TYPE])
            point.append(value)

        train_set_x_info.append(positive_point)
        train_set_x.append(point)
        train_set_y.append([1])

    for negative_point in negative_data:
        point = []
        for dim in negative_point:
            value = util.parse_value_with_data_type(dim[dn.VALUE], dim[dn.TYPE])
            point.append(value)

        train_set_x_info.append(negative_point)
        train_set_x.append(point)
        train_set_y.append([0])

    print("parsing finished")
    return train_set_x_info, train_set_x, train_set_y, variables, model_folder, model_file_name, point_number_limit


def generate_label_request(train_set_x, variables):
    output_list = []
    for point in train_set_x:
        tmp_list = []
        for dimension in point:
            index = point.index(dimension)
            var_name = variables[index].var_name
            var_type = variables[index].var_type

            tmp_dic = {dn.NAME: var_name}
            if var_type == dn.INTEGER:
                dimension = int(round(dimension))
            tmp_dic[dn.VALUE] = str(dimension)
            tmp_dic[dn.TYPE] = var_type

            tmp_list.append(tmp_dic)
        output_list.append(tmp_list)

    output_string = json.dumps(output_list)
    return output_string


def generate_point_info_request(train_set_x, variables):
    output_list = []
    for point in train_set_x:
        tmp_list = []
        for dimension in point:
            index = point.index(dimension)
            var_name = variables[index].var_name
            var_type = variables[index].var_type

            tmp_dic = {dn.NAME: var_name}
            if var_type == dn.INTEGER:
                dimension = int(round(dimension))
            tmp_dic[dn.VALUE] = str(dimension)
            tmp_dic[dn.TYPE] = var_type

            tmp_list.append(tmp_dic)
        output_list.append(tmp_list)

    output_string = json.dumps(output_list)
    return output_string

# parse_training_message_body(input)
# generate_label_request(test_data,"PRIMITIVE",["a","b"])
# label_input = {
#                  dn.RESULT: [[{dn.LABEL: True, dn.VALUE: 1, dn.TYPE: dn.INTEGER, dn.NAME: "a", "IS_PADDING": TRUE, "MODIFIABLE": TRUE},
#                            {dn.LABEL: True, dn.VALUE: 2, dn.TYPE: dn.INTEGER, dn.NAME: "b"}],
#                           [{dn.LABEL: True, dn.VALUE: 3, dn.TYPE: dn.INTEGER, dn.NAME: "a"},
#                            {dn.LABEL: True, dn.VALUE: 4, dn.TYPE: dn.INTEGER, dn.NAME: "b"}],
#                           [{dn.LABEL: False, dn.VALUE: 5, dn.TYPE: dn.INTEGER, dn.NAME: "a"},
#                            {dn.LABEL: False, dn.VALUE: 6, dn.TYPE: dn.INTEGER, dn.NAME: "b"}]]}

def parse_label(label_input):
    label_list = label_input[dn.RESULT]
    labels = []
    for point in label_list:
        label = point[0][dn.LABEL]
        if label:
            labels.append(1)
        else:
            labels.append(0)

    return labels

# parse_label(label_input)


# parse_training_message_body(input)
# generate_label_request(test_data,"PRIMITIVE",["a","b"])
# label_input = {
#                  dn.RESULT: [[{dn.VALUE: 1, dn.TYPE: dn.INTEGER, dn.NAME: "a", "IS_PADDING": TRUE, "MODIFIABLE": TRUE},
#                            {dn.VALUE: 2, dn.TYPE: dn.INTEGER, dn.NAME: "b"}],
#                           [{dn.VALUE: 3, dn.TYPE: dn.INTEGER, dn.NAME: "a"},
#                            {dn.VALUE: 4, dn.TYPE: dn.INTEGER, dn.NAME: "b"}],
#                           [{dn.VALUE: 5, dn.TYPE: dn.INTEGER, dn.NAME: "a"},
#                            {dn.VALUE: 6, dn.TYPE: dn.INTEGER, dn.NAME: "b"}]]}
def parse_point_info(data):
    label_list = data[dn.RESULT]
    info = []
    for point in label_list:
        dictionary = {dn.IS_PADDING: point[0][dn.IS_PADDING],
                      dn.MODIFIABLE: point[0][dn.MODIFIABLE]
                      }
        info.append(dictionary)

    return info
