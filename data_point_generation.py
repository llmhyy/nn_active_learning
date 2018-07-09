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
    trainName="train"+"_".join(str(x) for x in formu)+".csv"
    testName="test"+"_".join(str(x) for x in formu)+".csv"
    coefficientList=formu[:-1]
    y=formu[-1]
    with open(trainName, 'wb') as csvfile:
        
        train = csv.writer(csvfile)
        
        for k in range(700):

            #TODO coefficient and xList should comes from formu


            xList = []
            variableNum=len(coefficientList)
            for i in range(variableNum):
                
                xList.append(random.randint(-10,10))

            
            
            flag=tf.polynomialModel(coefficientList,xList,y)
            
            optList = []
            
            
                

            if (flag):
                optList.append(0.0)
                
                optList += xList

                train.writerow(optList)
            else:
                optList.append(1.0)
                                       
                optList += xList

                train.writerow(optList)

    testingPoint(formu,variableNum,400,-100,100,testName,formula.POLYNOMIAL)          
    return trainName,testName




def randomCircle(formu):
    number = random.randint(1, 20)
    number = 2
    print(number)

    with open('train_C.csv', 'wb') as csvfile:
        with open('test_C.csv', 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)
            for k in range(1000):
                xList = []
                circleList = []
                # for i in range(number):
                # xList.append(math.pow(random.uniform(-10, 10),power))
                circleList.append(random.uniform(-1.5, 1.5))
                circleList.append(random.uniform(-1.5, 1.5))
                coefficientList = []

                output = 0
                optList = []
                flag = tf.polycircleModel(formu[0], formu[1], circleList)
                if k < 700:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):

                    if (flag):
                        optList.append(0.0)
                        optList += circleList

                        train.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList += circleList

                        train.writerow(optList)
                else:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        optList.append(0.0)
                        optList += circleList

                        test.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList += circleList
                        test.writerow(optList)


def testingPoint(formu, dimension, number, lowerbound, largebound,path,category):
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
            
            if(category==formula.POLYHEDRON):
                flag = tf.polycircleModel(formu[0], formu[1], i)
                if (flag):
                    # point.append(0.0)
                    # for i in range(len)
                    i.insert(0, 0.0)
                else:
                    i.insert(0, 1.0)
                test.writerow(i)
            elif (category==formula.POLYNOMIAL):
                coefficientList=formu[:1]
                y=formu[-1]
                flag=tf.polynomialModel(coefficientList,i,y)
                optList=[]
                if (flag):
                    optList.append(0.0)
                    
                    optList +=i

                    test.writerow(optList)
                else:
                    optList.append(1.0)
                                           
                    optList += i

                    test.writerow(optList)



randomPolynomial([[1,2],[3],[4,5,6]])

