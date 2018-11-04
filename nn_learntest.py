import json
import traceback
from sys import stdin
from sys import stdout

import os
from main import label_tester as lt, mid_point_active_learning, json_handler

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 1000
mock = False
try:
    while True:
        request_type = stdin.readline()
        request_type = request_type.strip(" ").strip("\n")
        print("request type", request_type)
        message_body = stdin.readline()
        message_body = message_body.strip("\n")
        stdout.flush()
        print("message body", message_body)
        message_body = json.loads(message_body)

        if request_type == "$TRAINING":
            train_set_X, train_set_Y, variables, model_path = json_handler.parse_training_message_body(
                message_body)
            model_path = os.path.join("models", model_path)
            label_tester = lt.CoverageLabelTester(variables)
            mid_point_active_learning.generate_accuracy(train_set_X, train_set_Y,
                                                        learning_rate, training_epochs,
                                                        lower_bound, upper_bound, False,
                                                        label_tester, model_path)
        elif request_type == "$BOUNDARY_EXPLORATION":
            data_set, model_path = json_handler.parse_boundary_exploration(message_body)
            model_path = os.path.join("models", model_path)

        stdout.flush()
        print("finished!")
except Exception as e:
    print(e)
    traceback.print_exc()
    stdout.flush()
finally:
    # sess.close()
    print("finished!")

# import json
# from sys import stdin
# from sys import stdout
#
# print("start")
# i=0
#
# try:
#     while (1):
#         i = i + 1
#         request_type = stdin.readline()
#         request_type = request_type.strip(" ").strip("\n")
#         print("request", request_type)
#
#         message_body = stdin.readline()
#         message_body = message_body.strip(" ").strip("\n")
#         print("message", message_body)
#         # json.dump(data)
#         a = json.loads(message_body)
#
#         # print("@@PythonStart@@")
#         print(a)
#         # print("@@PythonEnd@@")
#
#         if (request_type == '$TRAINING' or request_type == '$SEND_LABEL'):
#             if i>5:
#                 print("$TRAINING_FINISH")
#             else:
#                 print("$REQUEST_LABEL")
#                 print(
#                     "[[{'VALUE': '1', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '2', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}], [{'VALUE': '3', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '4', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}], [{'VALUE': '5', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '6', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}]]")
#
#         stdout.flush()
# except Exception as e:
#     print(e)
# finally:
#     # sess.close()
#     print("finished!")
