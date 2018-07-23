from __future__ import print_function

import math
import random

import formula as f
import numpy as np
import tensorflow as tf
import boundary_remaining as br
import gradient_combination

import testing_function
import util

def append_large_gradient(sess, g, X, logits, formu, train_set_X, train_set_Y, catagory, pointsRatio, decision):
    new_train_set_X = []
    new_train_set_Y = []

    gradientList = g[0].tolist()

    g_list = []

    dimension = len(train_set_X[0])
    input_size = len(train_set_X)

    # print (type(gradientList))
    for i in range(input_size):
        grad = 0
        for j in range(dimension):
            grad += g[0][i][j] * g[0][i][j]
        g_list.append(math.sqrt(grad))
    util.quickSort(g_list)

    threshold = g_list[int(-len(gradientList) * pointsRatio)]

    print(threshold)
    for j in range(input_size):
        grad = 0
        new = br.decide_cross_boundry_point(sess, g, X, logits, train_set_X, j, threshold, decision)

        if (len(new)!=0):
            if (new not in train_set_X):
                print("ready to add new points: ", new)
                new_train_set_X.append(new)

                flag = testing_function.test_label(new, formu)
                if (flag):
                    new_train_set_Y.append([0])
                else:
                    new_train_set_Y.append([1])

    train_set_X = train_set_X + new_train_set_X
    train_set_Y = train_set_Y + new_train_set_Y

    return train_set_X, train_set_Y

def is_training_data_balanced(length_0, length_1, balance_ratio_threshold):
    return (length_0 / length_1 > balance_ratio_threshold and length_0 / length_1 < 1) \
           or \
           (length_1 / length_0 > balance_ratio_threshold and length_1 / length_0 < 1)

def generate_accuracy(train_path, test_path, formula, catagory):
    print("=========GRADIENT===========")

    # Parameters
    learning_rate = 0.1
    training_epochs = 100
    balance_ratio_threshold = 0.7
    display_step = 1
    changing_rate = [1000]
    step = 8
    pointsRatio = 0.25
    active_learning_iteration = 10

    train_set = []
    test_set_X = []
    test_set_Y = []
    train_set_X = []
    train_set_Y = []

    util.preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y, 
                    train_path, test_path, read_next=True)

    # Network Parameters
    n_hidden_1 = 10  # 1st layer number of neurons
    n_hidden_2 = 10  # 2nd layer number of neurons
    n_input = len(train_set_X[0])  # MNIST data input (img shape: 28*28)
    n_classes = 1  # MNIST total classes (0-9 digits)

    random_seed = 0
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
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

    train_acc_list = []
    test_acc_list = []
    result = []

    decision = gradient_combination.combination(n_input)

    for i in range(active_learning_iteration):
        print("*******", i, "th loop:")
        print("training set size", len(train_set_X))
        # ten times training
        with tf.Session() as sess:
            sess.run(init)
            label_0 = []
            label_1 = []

            label_0, label_1 = util.data_partition(train_set_X, train_set_Y)
            print(len(label_0), len(label_1))
            if (len(label_1) == 0 or len(label_0) == 0):
                raise Exception("Cannot be classified")

            g = sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})

            print(g)

            train_set_X, train_set_Y = append_large_gradient(sess, g, X, logits, formula, train_set_X, train_set_Y, catagory, pointsRatio, decision)

            length_0 = len(label_0) + 0.0
            length_1 = len(label_1) + 0.0

            print("label 0 length", length_0, "label 1 length", length_1)

            if (not is_training_data_balanced(length_0, length_1, balance_ratio_threshold)):
                br.apply_boundary_remaining(sess, newgrads, X, Y, length_0, length_1, logits, formula, train_set_X, train_set_Y)
            ##print(g)
            smallGradient_Unchanged = 0.0
            smallGradient_total = 0.0
            largeGradient_Unchanged = 0.0
            largeGradient_total = 0.0

            for epoch in range(training_epochs):
                _, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X, Y: train_set_Y})

            ##print(g)
            ##print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(c))
            # print(g)
            train_y = sess.run(logits, feed_dict={X: train_set_X})
            test_y = sess.run(logits, feed_dict={X: test_set_X})

            train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            test_acc = util.calculate_accuracy(test_y, test_set_Y, False)

            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)

            predicted = tf.cast(logits > 0.5, dtype=tf.float32)
            util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={X:x}), train_set_X, train_set_Y)

###########################################
# decide new points dimension by dimension
            # for j in range(len(train_set_X)):
            #     grad = 0
            #     for k in range(n_input):
            #         grad += g[0][j][k] * g[0][j][k]
            #     g_total = math.sqrt(grad)
            #     # print("Im here ==================================")
            #     new = []
            #     if (g_total > threshold):    
            #         for k in range(n_input):

            #             tmp1 = [x for x in train_set_X[j]]
            #             tmp1[k] = tmp1[k] + g[0][j][k] * (step / g_total)
            #             tmp2 = [x for x in train_set_X[j]]
            #             tmp2[k] = tmp2[k] - g[0][j][k] * (step / g_total)

            #             new_pointsX = [tmp1, tmp2, train_set_X[j]]
            #             new_pointsY = sess.run(logits, feed_dict={X: new_pointsX})
                       
            #             original_y = new_pointsY[-1]
            #             distances = [x for x in new_pointsY]
            #             distances = distances[:-1]
            #             # ans = 0
            #             if (original_y < 0.5):
            #                 ans = max(distances)
            #             else:
            #                 ans = min(distances)
            #             one_position = new_pointsX[distances.index(ans)]
            #             if (one_position==tmp1):
            #                 new.append(tmp1[k])
            #             else:
            #                 new.append(tmp2[k])

#############################################

                    
                ##boundary remaining test
                ##small gradient test
            #         X1=train_set_X[j][0]
            #         X2=train_set_X[j][1]
            #         newY=train_set_Y[j][0]
            #         g_x = g[0][j][0]
            #         g_y = g[0][j][1]
            #         g_total = math.sqrt(g_x*g_x+g_y*g_y)

            #         if (g_total==0) :
            #             tmpX1 = X1 - step
            #             tmpX2 = X2 + step
            #         else:
            #             tmpX1 = X1 - g[0][j][1]*(step/g_total)
            #             tmpX2 = X2 + g[0][j][0]*(step/g_total)
            #         ##print ("Y",newY)
            #         if(g[0][j][0]<0.01):

            #         	smallGradient_total+=1
            #         	if(newY==0):
            #         		if(polynomial_model(tmpX1,tmpX2)):
            #         			smallGradient_Unchanged+=1.0
            #         	elif(newY==1):
            #         		if(not polynomial_model(tmpX1,tmpX2)):
            #         			smallGradient_Unchanged+=1.0

            #         # ##large gradient test
            #         if(g[0][j][0]>0.01):
            #         	# newtmpX1=train_set_X[j][0]-g[0][j][0]*k
            #         	# newtmpX2=train_set_X[j][1]-g[0][j][1]*k

            #         	largeGradient_total+=1
            #         	if(newY==0):
            #         		if(polynomial_model(tmpX1,tmpX2)):
            #         			largeGradient_Unchanged+=1.0
            #         	elif(newY==1):
            #         		if(not polynomial_model(tmpX1,tmpX2)):
            #         			largeGradient_Unchanged+=1.0

            # # print("generated data points:")
            # for j in range(len(new_train_set_X)):
            #     print("(", new_train_set_X[j][0], ", ", new_train_set_X[j][1], ", ", new_train_set_X[j][2], ")", "label: ", new_train_set_Y[j][0])
            # if (smallGradient_total != 0) :
            #     print ("Small gradients", smallGradient_Unchanged/smallGradient_total)
            # if (largeGradient_total != 0):
            #     print ("Large gradients", largeGradient_Unchanged/largeGradient_total)
            # print(train_set_X)

    # print(smallGradient_total)
    # print (smallGradient_Unchanged)
    # print(largeGradient_total)
    # print (largeGradient_Unchanged)

    # print ("small gradient unchanged rate: ",smallGradient_Unchanged/smallGradient_total)
    # print ("large gradient unchanged rate: ", largeGradient_Unchanged/largeGradient_total)

    result.append(train_acc_list)
    result.append(test_acc_list)
    return result
