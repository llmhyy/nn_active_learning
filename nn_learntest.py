from sys import stdin
from sys import stdout

import json
import mid_point_active_learning
import json_handler
# print("start")

lower_bound = -1000
upper_bound = 1000

learning_rate = 0.01
training_epochs = 500
mock=False
try:
    while (1):
        data = stdin.readline()
        data = data.strip("\n")
        # json.dump(data)
        data= json.load(data)
        # print("@@PythonStart@@")
        # print(a)
        # print("@@PythonEnd@@")
        train_set_X,train_set_Y,name_list,type=json_handler.json_parser(data)
        if mock==False:
            inputX=train_set_X
            inputY=train_set_Y
            train_data_file=None
            test_data_file=None
            formu=None
            category=None

        mid_point_active_learning.generate_accuracy(inputX,inputY,train_data_file, test_data_file, formu, category, learning_rate, training_epochs, lower_bound, upper_bound,type,name_list,mock)
        stdout.flush()
        print("finished!")

finally:
# sess.close()
    print("finished!")

