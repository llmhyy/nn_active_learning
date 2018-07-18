from __future__ import print_function

import math
import random
from gradient import decide_gradient

import numpy as np
import tensorflow as tf
import formula

import testing_function
import util
import boundary_remaining as br

step = 3

def generate_accuracy(train_data_file, test_data_file,formu,category):

    # Parameters
    learning_rate = 1
    training_epochs = 100

    pointsNumber = 10
    active_learning_iteration = 10
    threhold = 5
    test_set_X = []
    test_set_Y = []
    train_set_X = []
    train_set_Y = []

    util.preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y, train_data_file,test_data_file,read_next=True)
    # Network Parameters
    n_hidden_1 = 10  # 1st layer number of neurons
    n_hidden_2 = 10  # 2nd layer number of neurons
    n_input = len(train_set_X[0])  # MNIST data input (img shape: 28*28)
    n_classes = 1  # MNIST total classes (0-9 digits)

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    
    train_acc_list=[]
    test_acc_list=[]
    result = []


    
    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0)),
        'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }


    # Construct model
    logits = util.multilayer_perceptron(X, weights, biases)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Initializing the variables
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    newgrads = tf.gradients(logits, X)
    y = None


    # class point_pair:
    #     def __init__(self, point_label_0, point_label_1, distance):
    #         self.point_label_0 = point_label_0
    #         self.point_label_1 = point_label_1
    #         self.distance = distance


    for i in range(active_learning_iteration):


        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))
        pointsNumber = 10
        with tf.Session() as sess:
            sess.run(init)
            label_0 = []
            label_1 = []
            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            print(len(label_0), len(label_1))
            
            if (len(label_1)==0 or len(label_0)==0):
                raise Exception("Cannot be classified")

            distanceList = []
            point_pairList = {}

            for m in label_0:
                for n in label_1:
                    distance=0
                    for d in range(n_input):

                        distance += (m[d] - n[d]) * (m[d] - n[d])
                    distance=math.sqrt(distance)
                    if (distance > threhold):

                        # if(distance in distanceList):
                        ##print ("cnm")
                        key=()
                        for h in range(n_input):
                            tmpKey=(m[h],n[h])
                            key=key+tmpKey
                       

                        value = distance
                        if (not point_pairList):
                            point_pairList[key] = value
                        elif (not (key in point_pairList.keys())):
                            point_pairList[key] = value
                        distanceList.append(distance)
                # print(m,n)

            # print (distanceList)

            util.quickSort(distanceList)
            # print(distanceList)
            selectedList = []
            # pivot=0
            # while pivot<pointsNumber:

            # 	if(distanceList[pivot] in selectedList):
            # 		pointsNumber+=1
            # 		pivot+=1

            # 	else:

            # 		selectedList.append(distanceList[pivot])
            # 		pivot+=1
            length = len(distanceList)
            index1 = length / 3
            index2 = length / 3 * 2
            pointer = 0
            for p in range(3):
                if (pointer < index1):
                    num = int(pointsNumber * 0.6)
                    util.addPoints(num, distanceList, selectedList, pointer)
                    pointer = index1
                elif (pointer < index2):
                    num = int(pointsNumber * 0.3)
                    util.addPoints(num, distanceList, selectedList, pointer)
                    #
                    pointer = index2
                else:
                    num = int(pointsNumber * 0.1)
                    util.addPoints(num, distanceList, selectedList, pointer)

                # pick large pooints

            ##print (selectedList)
            ##print (distanceList)
            # print (len(selectedList))
            # print (len(point_pairList))
            # print (train_set_X)


            for m in selectedList:
                for k, v in point_pairList.items():
                    if (m == v):
                        
                        point_0 = []
                        point_1 = []
                        for b in range(len(k)):
                            if(b%2==0):
                                point_0.append(k[b])
                            else:
                                point_1.append(k[b])
                        middlepoint = []
                        
                        for b in range(n_input):
                            middlepoint.append((point_0[b]+point_1[b])/2.0)
                        
                        # print (point_0)
                        # print("original point", point_0, point_1)
                        # print("middlepoint", middlepoint)
                      
                        if category == formula.POLYHEDRON:
                            flag = testing_function.polycircleModel(formu[0], formu[1], middlepoint)
                        elif category== formula.POLYNOMIAL:
                            flag= testing_function.polynomialModel(formu[:-1],middlepoint,formu[-1])
                       
                        if (flag):
                            if (middlepoint not in train_set_X):
                                train_set_X.append(middlepoint)
                                train_set_Y.append([0])

                        else:
                            if (middlepoint not in train_set_X):
                                train_set_X.append(middlepoint)
                                train_set_Y.append([1])


            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            length_0=len(label_0)+0.0
            length_1=len(label_1)+0.0
            
                            

            print ("label 0",length_0,"label 1",length_1)
            if (length_0/length_1>0.7 and length_0/length_1<1) or (length_1/length_0 >0.7 and length_1/length_0<1):
                for epoch in range(training_epochs):
                    _, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X, Y: train_set_Y})

                train_y = sess.run(logits, feed_dict={X: train_set_X})
                test_y = sess.run(logits, feed_dict={X: test_set_X})

                print("new train size after mid point", len(train_set_X), len(train_set_Y))
                train_acc = util.calculateAccuracy(train_y, train_set_Y, False)
                test_acc = util.calculateAccuracy(test_y, test_set_Y, False)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                continue

            label_selected=[]
            gradient_selected=[]
            length_added=0

            # compare if data is unbalanced
            label_flag=0
            g=sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})
            label_0, label_1,label_0_gradient,label_1_gradient = util.data_partition_gradient(train_set_X, train_set_Y,g[0])
            if length_0/length_1<0.7:
                label_selected=label_0
                gradient_selected=label_0_gradient
                length_added=length_1-length_0
                label_flag=0
            elif length_1/length_0<0.7:
                label_selected=label_1
                gradient_selected=label_1_gradient
                length_added=length_0-length_1
                label_flag=1
            else:
                continue                 
            print ("label 0",length_0,"label 1",length_1)

################################################################
# get all gradients for the unbalanced label points     

            #boundary remaining
            # print(g)
            gradient_list = []
            decision = decide_gradient(len(label_selected[0]))
            for j in range(len(label_selected)):
                grad = 0
                for k in range(len(label_selected[0])):
                    grad += gradient_selected[j][k] * gradient_selected[j][k]
                g_total = math.sqrt(grad)
                # print("Im here ==================================")
                if g_total==0:
                    tmpg = []
                    for d in range(len(label_selected[0])):
                        tmpg.append(1)
                        gradient_list.append(tmpg)
                    continue
                new_pointsX = []
                for k in range(len(decision)):
                    tmp = []
                    for h in range(len(label_selected[0])):
                        if (decision[k][h]==True):
                            tmp.append(label_selected[j][h] - gradient_selected[j][h] * (step / g_total))
                        else:
                            tmp.append(label_selected[j][h] + gradient_selected[j][h] * (step / g_total))
                    # tmp[k].append(train_set_X[j][k] + g[0][j][k] * (step / g_total))
                    new_pointsX.append(tmp)
                new_pointsX.append(label_selected[j])
                new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})

                original_y = new_pointsY[-1]
                distances = [x for x in new_pointsY]
                distances = distances[:-1]
                # ans = 0
                if (original_y < 0.5):
                    ans = min(distances)
                else:
                    ans = max(distances)
                direction = decision[distances.index(ans)]

                return_value = []
                for k in range(len(direction)):
                    if direction[k]==True:
                        return_value.append(-gradient_selected[j][k])
                    else:
                        return_value.append(gradient_selected[j][k])
                gradient_list.append(return_value)

################################################################                

            newX=br.balancingPoint(label_flag, label_selected,gradient_list,length_added, formu, category)
            counter=0
            for point in newX:
                if category == formula.POLYHEDRON:
                    flag = testing_function.polycircleModel(formu[0], formu[1], point)
                elif category== formula.POLYNOMIAL:
                    flag= testing_function.polynomialModel(formu[:-1],point,formu[-1])

                if (flag):
                    if label_flag==0:
                        counter+=1
                    train_set_X.append(point)
                    train_set_Y.append([0])

                else:
                    if label_flag==1:
                        counter+=1                    
                    train_set_X.append(point)
                    train_set_Y.append([1])                

            boundary_remaining_accuracy=(counter+0.0)/len(newX)
            print ("boundary remaining accuracy",boundary_remaining_accuracy)
            print ("new training size after boundary remaining",len(train_set_X),len(train_set_Y))
            for epoch in range(training_epochs):
                    _, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X, Y: train_set_Y})

            train_y = sess.run(logits, feed_dict={X: train_set_X})
            test_y = sess.run(logits, feed_dict={X: test_set_X})

            print("new train size after mid point", len(train_set_X), len(train_set_Y))
            train_acc = util.calculateAccuracy(train_y, train_set_Y, False)
            test_acc = util.calculateAccuracy(test_y, test_set_Y, False)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

    result.append(train_acc_list)
    result.append(test_acc_list)
    return result