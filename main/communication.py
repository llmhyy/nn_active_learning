from sys import stdout


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


def send_model_check_response(response_content):
    print("$MODEL_CHECK")
    print(response_content)
    stdout.flush()
