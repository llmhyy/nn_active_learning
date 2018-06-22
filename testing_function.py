import math


def logModel(x1, x2):
    if (x2 > math.log(x1)):
        return True
    else:
        return False


def circleModel(x1, x2):
    if (x1 * x1 + x2 * x2 > 100):
        return True
    else:
        return False


def polynomialModel(x1, x2):
    if (x2 > x1 * x1 * x1 + x1 * x1 + x1):
        return True
    else:
        return False
