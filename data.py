import csv
import random
import numpy as np
import math
import testing_function as tf
import models
from itertools import product
formula = models.formulas.get("circles",0)
def basic():
    with open('train.csv', 'wb') as csvfile:
        with open('test.csv', 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)
            for i in range(1000):
                x = random.uniform(-100, 100)
                y = random.uniform(-100, 100)
                flag = y > x * x * x + x * x + x
                if i < 700:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        train.writerow([0.0, x, y])
                    else:
                        train.writerow([1.0, x, y])
                else:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        test.writerow([0.0, x, y])
                    else:
                        test.writerow([1.0, x, y])

def randomPolynomial():
    number=random.randint(1,20)
    number = 2
    print (number)
    
    with open('trainP.csv', 'wb') as csvfile:
        with open('testP.csv', 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)
            coefficientList=[1]
            
            for k in range (1000):
                
                xList=[]
                out=[]
                for i in range(number-1):
                    #coefficientList.append(random.uniform(-0, 10))
                    power=number-i
                    x=random.uniform(-10, 10)
                    xList.append(x)
                    out.append(math.pow(x,power))
                    
                output=0
                for i in range (number-1):
                    output+=coefficientList[i]*out[i]
                y=random.uniform(0, 50)
                optList=[]
                xList.append(y)
                flag=y>output
                if k < 700:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):

                    if (flag):
                        optList.append(0.0)
                        optList+=xList

                        train.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList+=xList

                        train.writerow(optList)                                                
                else:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        optList.append(0.0)
                        optList+=xList

                        test.writerow(optList)                        
                    else:
                        optList.append(1.0)
                        optList+=xList
                        test.writerow(optList) 

def randomCircle():
    number=random.randint(1,20)
    number = 2
    print (number)
    
    with open('train_C.csv', 'wb') as csvfile:
        with open('test_C.csv', 'wb') as csvfile2:
            train = csv.writer(csvfile)
            test = csv.writer(csvfile2)
            for k in range (1000):
                xList=[]
                circleList=[]
                # for i in range(number):
                    # xList.append(math.pow(random.uniform(-10, 10),power))
                circleList.append(random.uniform(-1.5, 1.5))
                circleList.append(random.uniform(-1.5, 1.5))

                output=0
                optList=[]
                flag=tf.polycircleModel(formula[0],formula[1],circleList)
                if k < 700:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):

                    if (flag):
                        optList.append(0.0)
                        optList+=circleList

                        train.writerow(optList)
                    else:
                        optList.append(1.0)
                        optList+=circleList

                        train.writerow(optList)                                                
                else:
                    ##if ((x-12.5)*(x-12.5)+y*y<100 or (x+12.5)*(x+12.5)+y*y<100):
                    if (flag):
                        optList.append(0.0)
                        optList+=circleList

                        test.writerow(optList)                        
                    else:
                        optList.append(1.0)
                        optList+=circleList
                        test.writerow(optList) 

def testingPoint(dimension,number,lowerbound,largebound):
    with open('test_C.csv', 'wb') as csvfile:
        numberOfPoint=int(round(math.pow(number,(1.0/dimension))))
        step=(largebound-lowerbound)/float(numberOfPoint)
        pointList=[]
        for i in range(numberOfPoint):
            pointList.append(lowerbound+i*step)

        output=list(product(pointList,repeat=dimension))
        test = csv.writer(csvfile)
        for i in output:
            i = list(i)
            point = []
            flag = tf.polycircleModel(formula[0],formula[1],i)
            if (flag):
                # point.append(0.0)
                # for i in range(len)
                i.insert(0, 0.0)
            else:
                i.insert(0, 1.0)
            test.writerow(i)
randomCircle()
testingPoint(2,4000,-1.5,1.5)