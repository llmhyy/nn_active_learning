from sys import stdout


def send_training_finish_message():
    print("$TRAINING_FINISH")
    stdout.flush()


def send_label_request(request_string):
    print("$REQUEST_LABEL")
    print(request_string)
    stdout.flush()
