import math
import formula
import json_handler
from sys import stdin
from sys import stdout

def logModel(x1, x2):
    if (x2 > math.log(x1)):
        return True
    else:
        return False


def circleModel(x1, x2):
    if ((x1-12.5)*(x1-12.5)+x2*x2<100 or (x1+12.5)*(x1+12.5)+x2*x2<100):
        return True
    else:
        return False


def polynomial_model(coefficientList, x, y):

    variableNum=len(coefficientList)
    output=0
    for i in range(variableNum):
        tmpList=coefficientList[i]
        tmpLength=len(tmpList)
        for j in range(tmpLength):
            power=tmpLength-j
                
            output+=tmpList[j]*math.pow(x[i],power)
    if (output>y):
        return True
    else:
        return False

def polycircle_model(center, radius, x):
    # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
    for i in range(len(center)):
        point_radius = 0
        for j in range(len(x)):
            point_radius += (x[j]-center[i][j])*(x[j]-center[i][j])
        if (point_radius < radius[i]*radius[i]):
            return True
    return False


def test_label(points, formu,train_set_X,train_set_Y,type,name_list,mock):
    if mock==True:
        category = formu.get_category()
        form = formu.get_list()
        flag = True
        if category == formula.POLYHEDRON:
            for point in points:
                flag = polycircle_model(form[0], form[1], point)
                if (flag):
                    train_set_X.append(point)
                    train_set_Y.append([1])
                    print ("added point: ",point,flag)
                else:
                    train_set_X.append(point)
                    train_set_Y.append([0])
                    print ("added point: ",point,flag)
        elif category == formula.POLYNOMIAL:
            for point in points:
                flag = polynomial_model(form[:-1], point, form[-1])
                if (flag):
                    train_set_X.append(point)
                    train_set_Y.append([1])
                    print ("added point: ",point,flag)
                else:
                    train_set_X.append(point)
                    train_set_Y.append([0])
        return train_set_X,train_set_Y

    else:
        json_handler.requestLabel(points,type,name_list)
        data = stdin.readline()
        data = data.strip("\n")
        newX,newY,name_list=json_handler.json_parser(data)
        train_set_X=train_set_X+newX
        train_set_Y=train_set_Y+newY
        return train_set_X,train_set_Y