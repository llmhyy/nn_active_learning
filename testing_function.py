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


def polynomialModel(coefficient,x):

    number=2
    y=x[-1]
    output=0
    for i in range (number-1):
        power=number-i
        output+=coefficient[i]*math.pow(x[i],power) 
    return y>output


def polycircleModel(center, radius, x):  
    # center format:[[0,1,3],[1,1,1]], radius format: [10,25], x format: [1,2,3]
    for i in range(len(center)):
        pointradius = 0
        for j in range(len(x)):
            pointradius += (x[j]-center[i][j])*(x[j]-center[i][j])
        if (pointradius < radius[i]*radius[i]):
            return True
    return False