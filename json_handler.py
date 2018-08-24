import json
from sys import stdout

input = {"BRANCH_ID": "2-3",

         "POSITIVE_DATA": [
             [{"VALUE": "1", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "1", "TYPE": "PRIMITIVE", "NAME": "b"}]],

         "NEGATIVE_DATA": [
             [{"VALUE": "877", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "0", "TYPE": "PRIMITIVE", "NAME": "b"}],
             [{"VALUE": "548", "TYPE": "PRIMITIVE", "NAME": "a"}, {"VALUE": "969", "TYPE": "PRIMITIVE", "NAME": "b"}]
         ]}

test_data = [[1, 2], [3, 4], [5, 6]]


def json_parser(input):
    print("starting parse data from java")

    train_set_X = []
    train_set_Y = []
    name_list = []
    positive_data = input["POSITIVE_DATA"]
    negative_data = input["NEGATIVE_DATA"]
    type = None
    for points in positive_data:
        tmp = []
        tmp_name = []
        for point in points:
            type = point["TYPE"]
            if type == "INTEGER":
                tmp.append(int(point["VALUE"]))
            tmp_name.append(point["NAME"])

        if name_list == []:
            name_list = tmp_name
        train_set_X.append(tmp)
        train_set_Y.append([1])

    for points in negative_data:
        tmp = []
        tmp_name = []
        for point in points:
            type = point["TYPE"]
            if type == "INTEGER":
                tmp.append(int(point["VALUE"]))
            tmp_name.append(point["NAME"])

        if name_list == []:
            name_list = tmp_name
        train_set_X.append(tmp)
        train_set_Y.append([0])

    print(train_set_X)
    print(train_set_Y)
    print(name_list)
    print("parsing finished")
    return train_set_X, train_set_Y, name_list, type


def requestLabel(train_set_X, type, name_list):
    outputList = []
    for point in train_set_X:
        tmp_list = []
        for coordinate in point:
            tmp_dic = {}
            tmp_dic["NAME"] = name_list[point.index(coordinate)]
            if type == "INTEGER":
                coordinate = int(coordinate)
            tmp_dic["VALUE"] = str(coordinate)
            tmp_dic["TYPE"] = type

            tmp_list.append(tmp_dic)
        outputList.append(tmp_list)

    outputString = json.dumps(outputList)
    print("$REQUEST_LABEL")
    print(outputString)
    stdout.flush()

    # print ("finish sending")

    return outputString


# json_parser(input)
# requestLabel(test_data,"PRIMITIVE",["a","b"])
label_input = {"RESULT": [[{"LABEL": True, "VALUE": 1, "TYPE": "INTEGER", "NAME": "a"},
                           {"LABEL": True, "VALUE": 2, "TYPE": "INTEGER", "NAME": "b"}],
                          [{"LABEL": True, "VALUE": 3, "TYPE": "INTEGER", "NAME": "a"},
                           {"LABEL": True, "VALUE": 4, "TYPE": "INTEGER", "NAME": "b"}],
                          [{"LABEL": False, "VALUE": 5, "TYPE": "INTEGER", "NAME": "a"},
                           {"LABEL": False, "VALUE": 6, "TYPE": "INTEGER", "NAME": "b"}]]}


def label_parser(input):
    label_list = input["RESULT"]
    output = []
    for point in label_list:

        label = point[0]["LABEL"]
        if label == True:
            output.append(1)
        else:
            output.append(0)
    print(output)
    stdout.flush()
    return output

# label_parser(label_input)
