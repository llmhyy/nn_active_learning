import csv
import math
import random
from itertools import product

import formula
import testing_function as tf

def generate_data_points(formu, category):
    if (category == formula.POLYHEDRON):
        trainpath, testpath = randomCircle(formu)
    elif (category == formula.POLYNOMIAL):
        trainpath, testpath = randomPolynomial(formu)
    return trainpath, testpath

#TODO coefficient and xList should comes from formu
def randomPolynomial(formu):
    trainName="train"+"_".join(str(x) for x in formu)+".csv"
    testName="test"+"_".join(str(x) for x in formu)+".csv"

    train_path = "./dataset/"+trainName
    test_path = "./dataset/"+testName

    coefficientList=formu[:-1]
    y=formu[-1]
    with open(train_path, 'wb') as csvfile:
        
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

    testingPoint(formu,variableNum,400,-10,10,test_path,formula.POLYNOMIAL)          
    return train_path,test_path



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
    return train_path, test_path

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
            else:
                flag = tf.polynomialModel(formu[:-1],i,formu[-1])

            if (flag):
                i.insert(0, 0.0)
            else:
                i.insert(0, 1.0)
            test.writerow(i)

# testingPoint(2, 4000, -1.5, 1.5)
# randomPolynomial([[1,2],[3],[4,5,6]])

