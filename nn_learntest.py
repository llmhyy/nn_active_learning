import json
import os
import traceback
from sys import stdin
from sys import stdout

import tensorflow as tf

from main import util, label_tester as lt, mid_point_active_learning, json_handler, communication, boundary_exploration as be

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

        tf.reset_default_graph()

        if request_type == "$TRAINING":
            train_set_x_info, train_set_x, train_set_y, variables, model_folder, model_file_name, point_number_limit \
                = json_handler.parse_training_message_body(message_body)
            model_folder = os.path.join("models", model_folder)
            label_tester = lt.CoverageLabelTester(variables)

            train_set_x = util.convert_with_mask(train_set_x, train_set_x_info)
            mid_point_learner = mid_point_active_learning.MidPointActiveLearner(train_set_x_info, train_set_x,
                                                                                train_set_y, None, None,
                                                                                learning_rate, training_epochs,
                                                                                lower_bound, upper_bound, False,
                                                                                label_tester,
                                                                                point_number_limit, model_folder,
                                                                                model_file_name)
            mid_point_learner.generate_accuracy()
        elif request_type == "$BOUNDARY_EXPLORATION":
            data_set_info, data_set, model_folder, model_file_name, variables = json_handler.parse_boundary_exploration(
                message_body)
            data_set = util.convert_with_mask(data_set, data_set_info)
            model_folder = os.path.join("models", model_folder)
            label_tester = lt.CoverageLabelTester(variables)

            boundary_explorer = be.BoundaryExplorer(data_set_info, data_set, model_folder, model_file_name,
                                                    label_tester, 3)
            boundary_explorer.boundary_explore()
            communication.send_exploration_finish_message()
        elif request_type == "$MODEL_CHECK":
            branch_id, method_id = json_handler.parse_model_check(message_body)
            model_folder = os.path.join("models", method_id)
            model_folder = os.path.join(model_folder, branch_id)
            model_folder = os.path.join(model_folder, branch_id + ".meta")
            response_type = "$MODEL_CHECK"
            response_content = "FALSE"
            if os.path.exists(model_folder):
                response_content = "TRUE"
            message_string = json_handler.generate_model_check_response(response_content)
            communication.send_model_check_response(message_string)

        stdout.flush()
        print("finished!")
except Exception as e:
    print(e)
    traceback.print_exc()
    stdout.flush()
finally:
    # sess.close()
    print("finished!")
