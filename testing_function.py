import math


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


def polynomialModel(coefficientList,x,y):

    variableNum=len(coefficientList)
    output=0
    for i in range(variableNum):
        tmpList=coefficientList[i]
        tmpLength=len(tmpList)
        for j in range(tmpLength):
            power=tmpLength-j
                
            output+=tmpList[j]*math.pow(x[i],power)
            print ("output",output)


def polycircleModel(center, radius, x):  
    # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
    for i in range(len(center)):
        pointradius = 0
        for j in range(len(x)):
            pointradius += (x[j]-center[i][j])*(x[j]-center[i][j])
        if (pointradius < radius[i]*radius[i]):
            return True
    return False