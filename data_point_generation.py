import csv
import math
import random
from itertools import product

import formula
import testing_function as tf

def generate_data_points(formu, category):
    if (category == formula.POLYHEDRON):
        randomCircle(formu)
    elif (category == formula.POLYNOMIAL):
        randomPolynomial(formu)

#TODO coefficient and xList should comes from formu
def randomPolynomial(formu):
    number = random.randint(1, 20)

    print(number)
    with open('trainP.csv', 'wb') as csvfile:
        with open('testP.csv', 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)
            for k in range(1000):

                #TODO coefficient and xList should comes from formu

                coefficientList = []
                xList = []

                for i in range(number):
                    coefficientList.append(random.uniform(-0, 10))
                    power = number - i
                    xList.append(math.pow(random.uniform(-10, 10), power))

                output = 0
                for i in range(number):
                    output += coefficientList[i] * xList[i]
                y = random.uniform(-100000, 100000)
                optList = []

                flag = y > output
                if k < 700:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):

                    if (flag):
                        optList.append(0.0)
                        optList += xList

                        train.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList += xList

                        train.writerow(optList)
                else:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        optList.append(0.0)
                        optList += xList

                        test.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList += xList
                        test.writerow(optList)

# generate random data points for a circle formula
def randomCircle(formu):  # [[[12,0],[-12,0]],[4,4]]
    number = random.randint(1, 20)
    dim = len(formu[0][0])
    print(dim)

    trainname = "train" + "_".join(str(x) for x in formu[1]) + ".csv"
    testname = "test" + "_".join(str(x) for x in formu[1]) + ".csv"

    train_path = "./dataset/"+trainname
    test_path = "./dataset/"+testname

    with open(train_path, 'wb') as csvfile:
        with open(test_path, 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)

            for k in range(700):
                data_point = []
                generated_point = []
                for i in range(dim):
                    generated_point.append(random.uniform(-1000, 1000))

                flag = tf.polycircleModel(formu[0], formu[1], generated_point)

                if (flag):
                    data_point.append(0.0)
                    data_point += generated_point

                    train.writerow(data_point)
                else:
                    data_point.append(1.0)
                    data_point += generated_point

                    train.writerow(data_point)
            testingPoint(formu, dim, 4000, -1000, 1000, test_path, formula.POLYHEDRON)


def testingPoint(formu, dimension, number, lowerbound, largebound, path, catagory):
    with open(path, 'wb') as csvfile:
        numberOfPoint = int(round(math.pow(number, (1.0 / dimension))))
        step = (largebound - lowerbound) / float(numberOfPoint)
        pointList = []
        for i in range(numberOfPoint):
            pointList.append(lowerbound + i * step)

        output = list(product(pointList, repeat=dimension))
        test = csv.writer(csvfile)
        for i in output:
            i = list(i)

            if catagory==formula.POLYHEDRON:
                flag = tf.polycircleModel(formu[0], formu[1], i)
            # else:
            #     flag = 

            if (flag):
                i.insert(0, 0.0)
            else:
                i.insert(0, 1.0)
            test.writerow(i)

# testingPoint(2, 4000, -1.5, 1.5)
