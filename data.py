import csv
import random
import numpy as np
import math
import testing_function as tf
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
            for k in range (1000):
                coefficientList=[]
                xList=[]
                circleList=[]
                for i in range(number):
                    coefficientList.append(random.uniform(-0, 10))
                    power=number-i
                    xList.append(math.pow(random.uniform(-10, 10),power))
                    circleList.append(random.uniform(-2, 2))
                output=0
                for i in range (number):
                    output+=coefficientList[i]*xList[i]
                y=random.uniform(-100000, 100000)
                optList=[]
                flag=tf.polycircleModel([[1,1],[-1,-1]],[1,1],circleList)
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


randomPolynomial()