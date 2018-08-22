import json
from sys import stdin
from sys import stdout

print("start")
i=0

try:
    while (1):
        i = i + 1
        request_type = stdin.readline()
        request_type = request_type.strip(" ").strip("\n")
        print("request", request_type)

        message_body = stdin.readline()
        message_body = message_body.strip(" ").strip("\n")
        print("message", message_body)
        # json.dump(data)
        a = json.loads(message_body)

        # print("@@PythonStart@@")
        print(a)
        # print("@@PythonEnd@@")

        if (request_type == '$TRAINING' or request_type == '$SEND_LABEL'):
            if i>5:
                print("$TRAINING_FINISH")
            else:
                print("$REQUEST_LABEL")
                print(
                    "[[{'VALUE': '1', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '2', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}], [{'VALUE': '3', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '4', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}], [{'VALUE': '5', 'TYPE': 'PRIMITIVE', 'NAME': 'a'}, {'VALUE': '6', 'TYPE': 'PRIMITIVE', 'NAME': 'b'}]]")

        stdout.flush()
except Exception as e:
    print(e)
finally:
    # sess.close()
    print("finished!")
