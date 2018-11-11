from sys import stdout
from main import json_handler


def send_training_finish_message():
    print("$TRAINING_FINISH")
    stdout.flush()


def send_exploration_finish_message():
    print("$EXPLORATION_FINISH")
    stdout.flush()


def send_label_request(request_string):
    print("$REQUEST_LABEL")
    print(request_string)
    stdout.flush()


def send_point_info_request(request_string):
    print("$REQUEST_MASK_RESULT")
    print(request_string)
    stdout.flush()


def send_model_check_response(response_content):
    print("$MODEL_CHECK")
    print(response_content)
    stdout.flush()


def send_boundary_remaining_points(new_point_list, variables):
    message = json_handler.generate_boundary_remaining_message(new_point_list, variables)
    print("$SEND_BOUNDARY_REMAINING_POINTS")
    print(message)
    stdout.flush()
