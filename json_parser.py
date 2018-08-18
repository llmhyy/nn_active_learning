input={"BRANCH_ID":"2-3",

"POSITIVE_DATA":[[{"VALUE":"1","TYPE":"PRIMITIVE","NAME":"a"},{"VALUE":"1","TYPE":"PRIMITIVE","NAME":"b"}]],

"NEGATIVE_DATA":[
[{"VALUE":"877","TYPE":"PRIMITIVE","NAME":"a"},{"VALUE":"0","TYPE":"PRIMITIVE","NAME":"b"}],
[{"VALUE":"548","TYPE":"PRIMITIVE","NAME":"a"},{"VALUE":"969","TYPE":"PRIMITIVE","NAME":"b"}]
]}

def json_parser(input):
    train_set_X=[]
    train_set_Y=[]

    positive_data=input["POSITIVE_DATA"]
    negative_data=input["NEGATIVE_DATA"]

    for points in positive_data:
        tmp=[]
        for point in points:
            tmp.append(float(point["VALUE"]))

        train_set_X.append(tmp)
        train_set_Y.append([1])

    for points in negative_data:
        tmp = []
        for point in points:
            tmp.append(float(point["VALUE"]))

        train_set_X.append(tmp)
        train_set_Y.append([0])

    print (train_set_X)
    print (train_set_Y)


json_parser(input)



