""" Multilayer Perceptron.
A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).
Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import csv
import numpy as np
import random

# Parameters
learning_rate = 1


training_epochs = 10
display_step = 1

# Network Parameters
n_hidden_1 = 10  # 1st layer number of neurons
n_hidden_2 = 10  # 2nd layer number of neurons
n_input = 2  # MNIST data input (img shape: 28*28)
n_classes = 1  # MNIST total classes (0-9 digits)

random_seed = 0
random.seed(random_seed)
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
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

train_set = []
test_set_X = []
test_set_Y = []
train_set_X = []
train_set_Y = []

# read training data
with open('train.csv', 'rt') as csvfile:
    with open('train_next.csv', 'wb') as file:
        i=0
        spamreader = csv.reader(csvfile)
        writer = csv.writer(file)
        for row in spamreader:
            if(i>70):
                break
            i+=1
            
            writer.writerow(row)
            writer.writerow([1-float(row[0]),float(row[1])+0.01,float(row[2])+0.01])
            
            
    file.close()

# read testing data
with open('test.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        # print(row)
        l = [0, 0]
        l[0] = float(row[1])
        l[1] = float(row[2])
        # print(l)
        test_set_X.append(l)
        if (row[0] == '1.0'):
            test_set_Y.append([1])
        else:
            test_set_Y.append([0])

# read testing data
with open('train.csv', 'rt') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        l = [0, 0]
        l[0] = float(row[1])
        l[1] = float(row[2])
        # print(l)
        train_set_X.append(l)
        if (row[0] == '1.0'):
            train_set_Y.append([1])
        else:
            train_set_Y.append([0])


def calculateAccuracy(y, set_Y, b):
    test_correct = []
    test_wrong = []
    train_correct = []
    train_wrong = []
    for i in range(len(set_Y)):
        if (b):
            print(i, " predict:", y[i][0], " actual: ", set_Y[i][0])

        if y[i][0] > 0.5 and set_Y[i][0] == 1:
            test_correct.append(y[i])
        elif y[i][0] > 0.5 and set_Y[i][0] == 0:
            test_wrong.append(y[i])
        elif y[i][0] <= 0.5 and set_Y[i][0] == 0:
            test_correct.append(y[i])
        else:
            test_wrong.append(y[i])
    # print (test_correct)
    # print (test_wrong)

    print(len(test_correct) / float(len(test_correct) + len(test_wrong)))


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    ##x0=tf.nn.batch_normalization(x,mean=0.01, variance=1,offset=0,scale=1,variance_epsilon=0.001)
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # layer1_out = tf.sigmoid(layer_1)

    # Hidden fully connected layer with 256 neurons
    # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # layer2_out = tf.sigmoid(layer_2)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return tf.nn.sigmoid(out_layer)


# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# Initializing the variables
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

grads = tf.gradients(loss_op, weights["out"])
newgrads = tf.gradients(logits, X)

y = None

with tf.Session() as sess:
    sess.run(init)

    h1 = sess.run(weights["h1"])
    out = sess.run(weights["out"])

    print("h1", h1)
    print("out", out)

    g = sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})
    ##print(g)

    ##global gradients                                                        Y: train_set_Y}
    # Training cycle
    for epoch in range(training_epochs):

        _, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X,
                                                        Y: train_set_Y})

        g = sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})
        ##print(g)
        print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(c))

    print("Optimization Finished!")
    print(len(train_set_Y))
    train_y = sess.run(logits, feed_dict={X: train_set_X})
    test_y = sess.run(logits, feed_dict={X: test_set_X})

    # print(len(train_y))
    # print(len(train_set_Y))
    calculateAccuracy(train_y, train_set_Y, False)
    calculateAccuracy(test_y, test_set_Y, False)
    
    ##print (y)
    # y=y[0]
    # print (test_set_Y)

    # correct_prediction = tf.equal + len(test_wrong(tf.argmax(logits, 1), tf.argmax(Y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print("Test Accuracy:", accuracy.eval({X: np.asarray(test_set_X), Y: np.asarray(test_set_Y)}))
    # print("train Accuracy:", accuracy.eval({X: np.asarray(train_set_X), Y: np.asarray(train_set_Y)}))

# ##update csv
# with open('train_next.csv','a') as file:
# 	print('xixi')
# 	writer=csv.writer(file)
# 	print(len(gradients[0]))
# 	for i in range (len(gradients[0])):
# 		##print (float(gradients[0][i][0]))

# 		if((gradients[0][i][0])!=0.0):
# 			print('haha')
# 			writer.writerow(gradients[0][i])
