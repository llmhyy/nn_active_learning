import json
import traceback
from sys import stdin
from sys import stdout

import json_handler
import mid_point_active_learning

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 500
mock = False
try:
    while (True):
        request_type = stdin.readline()
        request_type = request_type.strip(" ").strip("\n")
        print("request type", request_type)
        message_body = stdin.readline()
        message_body = message_body.strip("\n")
        print("message body", message_body)
        stdout.flush()
        # json.dump(data)
        message_body = json.loads(message_body)
        # print("@@PythonStart@@")
        # print(a)
        # print("@@PythonEnd@@")
        train_set_X, train_set_Y, name_list, type = json_handler.json_parser(message_body)
        if mock == False:
            inputX = train_set_X
            inputY = train_set_Y
            train_data_file = None
            test_data_file = None
            formu = None
            category = None

        mid_point_active_learning.generate_accuracy(inputX, inputY, train_data_file, test_data_file, formu, category,
                                                    learning_rate, training_epochs, lower_bound, upper_bound, False, type,
                                                    name_list, mock)

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
