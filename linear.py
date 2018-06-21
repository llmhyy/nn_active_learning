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
import util

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

util.preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y)

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
    util.calculateAccuracy(train_y, train_set_Y, False)
    util.calculateAccuracy(test_y, test_set_Y, False)
    
