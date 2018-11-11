import json
import os
from sys import stdout

from main import variable as v


# input = {
#          "METHOD_ID": "com.test.Class.method()",
#          "BRANCH_ID": "2-3",
#          "DATA": [
#              [{"VALUE": "877", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "0", "TYPE": "PRIMITIVE", "NAME": "b"}],
#              [{"VALUE": "548", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "969", "TYPE": "PRIMITIVE", "NAME": "b"}]
#          ]}
#
# test_data = [[1, 2], [3, 4], [5, 6]]
def parse_boundary_exploration(message):
    model_folder = message["METHOD_ID"]
    model_file_name = message["BRANCH_ID"]

    model_path = os.path.join(model_folder, model_file_name)

    data = message["DATA"]

    sample_point = data[0]
    variables = []
    for dimension in sample_point:
        variable = v.Variable(dimension["NAME"], dimension["TYPE"])
        variables.append(variable)

    data_set = []
    for point in data:
        point = []
        for dimension in point:
            point.append(int(dimension["VALUE"]))

        data_set.append(point)

    return data_set, model_path


# input = {
#           "METHOD_ID": "com.test.Class.method()",
#           "BRANCH_ID": "2-3",
#
#          "POSITIVE_DATA": [
#              [{"VALUE": "1", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "1", "TYPE": "PRIMITIVE", "NAME": "b"}]],
#
#          "NEGATIVE_DATA": [
#              [{"VALUE": "877", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "0", "TYPE": "PRIMITIVE", "NAME": "b"}],
#              [{"VALUE": "548", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "969", "TYPE": "PRIMITIVE", "NAME": "b"}]
#          ]}
#
# test_data = [[1, 2], [3, 4], [5, 6]]
def parse_training_message_body(message):
    model_folder = message["METHOD_ID"]
    model_file_name = message["BRANCH_ID"]
    model_path = os.path.join(model_folder, model_file_name)

    train_set_X = []
    train_set_Y = []

    positive_data = message["POSITIVE_DATA"]
    negative_data = message["NEGATIVE_DATA"]

    sample_point = positive_data[0]
    variables = []
    for dimension in sample_point:
        variable = v.Variable(dimension["NAME"], dimension["TYPE"])
        variables.append(variable)

    for points in positive_data:
        tmp_point = []
        for point in points:
            if type == "INTEGER":
                tmp_point.append(int(point["VALUE"]))

        train_set_X.append(tmp_point)
        train_set_Y.append([1])

    for points in negative_data:
        tmp_point = []
        for point in points:
            if type == "INTEGER":
                tmp_point.append(int(point["VALUE"]))

        train_set_X.append(tmp_point)
        train_set_Y.append([0])

    # print(train_set_X)
    # print(train_set_Y)
    print("parsing finished")
    return train_set_X, train_set_Y, variables, model_path


def generate_label_request(train_set_X, variables):
    output_list = []
    for point in train_set_X:
        tmp_list = []
        for dimension in point:
            index = point.index(dimension)
            var_name = variables[index].var_name
            var_type = variables[index].var_type

            tmp_dic = {"NAME": var_name}
            if var_type == "INTEGER":
                dimension = int(round(dimension))
            tmp_dic["VALUE"] = str(dimension)
            tmp_dic["TYPE"] = var_type

            tmp_list.append(tmp_dic)
        output_list.append(tmp_list)

    output_string = json.dumps(output_list)
    return output_string


# parse_training_message_body(input)
# generate_label_request(test_data,"PRIMITIVE",["a","b"])
# label_input = {
#                  "RESULT": [[{"LABEL": True, "VALUE": 1, "TYPE": "INTEGER", "NAME": "a"},
#                            {"LABEL": True, "VALUE": 2, "TYPE": "INTEGER", "NAME": "b"}],
#                           [{"LABEL": True, "VALUE": 3, "TYPE": "INTEGER", "NAME": "a"},
#                            {"LABEL": True, "VALUE": 4, "TYPE": "INTEGER", "NAME": "b"}],
#                           [{"LABEL": False, "VALUE": 5, "TYPE": "INTEGER", "NAME": "a"},
#                            {"LABEL": False, "VALUE": 6, "TYPE": "INTEGER", "NAME": "b"}]]}

def parse_label(label_input):
    label_list = label_input["RESULT"]
    output = []
    for point in label_list:

        label = point[0]["LABEL"]
        if label:
            output.append(1)
        else:
            output.append(0)
    print(output)
    stdout.flush()
    return output

# parse_label(label_input)
