from __future__ import print_function

import random
import math

import numpy as np
import tensorflow as tf

import testing_function
import util

# Parameters
learning_rate = 1
training_epochs = 100



pointsNumber=20
active_learning_iteration = 7
threhold=8

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

test_set_X = []
test_set_Y = []
train_set_X = []
train_set_Y = []



util.preprocess(train_set_X, train_set_Y, test_set_X, test_set_Y, 'train_next.csv')

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


class point_pair:
	def __init__(self,point_label_0,point_label_1,distance):
		self.point_label_0=point_label_0
		self.point_label_1=point_label_1
		self.distance=distance







for i in range(active_learning_iteration):
	print("*******", i, "th loop:")
	print("training set size", len(train_set_X))
	pointsNumber=20
	with tf.Session() as sess:
		sess.run(init)
		label_0=[]
		label_1=[]
		label_0,label_1=util.data_partition(train_set_X,train_set_Y)
		print (len(label_0),len(label_1))
		distanceList=[]
		point_pairList=[]

		for m in label_0:
			for n in label_1:

				distance=math.sqrt((m[0]-n[0])*(m[0]-n[0])+(m[1]-n[1])*(m[1]-n[1]))
				if(distance>15):

					#if(distance in distanceList):
						##print ("cnm")
					pair=point_pair(m,n,distance)
					if(len(point_pairList)==0):
						point_pairList.append(pair)
					else:						
						for k in point_pairList:
							if(not (k.point_label_0==m and k.point_label_1==n)):

								point_pairList.append(pair)
					distanceList.append(distance)
					#print(m,n)
		
		#print (len(distanceList))


		util.quickSort(distanceList)
		selectedList=[]
		pivot=0
		while pivot<pointsNumber:
			
			if(distanceList[pivot] in selectedList):
				pointsNumber+=1
				pivot+=1

			else:

				selectedList.append(distanceList[pivot])
				pivot+=1
			
		print (selectedList)
		print (len(selectedList))
		print (len(point_pairList))
		#print (train_set_X)
		for m in selectedList:
			for n in point_pairList:
				if(m==n.distance):

					point_0=n.point_label_0
					point_1=n.point_label_1
					middlepoint=[]
					middlepoint.append((point_0[0]+point_1[0])/2.0)
					middlepoint.append((point_0[1]+point_1[1])/2.0)
					print (point_0)
					print ("original point",point_0,point_1)
					print("middlepoint",middlepoint)	
					flag=testing_function.polynomialModel(middlepoint[0],middlepoint[1])
					if(flag):
						train_set_X.append(middlepoint)
						train_set_Y.append([0])
						# print ( train_set_X.index(point_0))
						# print(point_0)
						# train_set_X.remove(train_set_X[train_set_X.index(point_0)])

					else:
						train_set_X.append(middlepoint)
						train_set_Y.append([1])
						# print (train_set_X.index(point_1))
						# print(point_1)
											
						# train_set_X.remove(train_set_X[train_set_X.index(point_1)])
		for epoch in range(training_epochs):
			_, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X, Y: train_set_Y})				
		train_y = sess.run(logits, feed_dict={X: train_set_X})
		test_y = sess.run(logits, feed_dict={X: test_set_X})

		print("new train size",len(train_set_X),len(train_set_Y))
		util.calculateAccuracy(train_y, train_set_Y, False)
		util.calculateAccuracy(test_y, test_set_Y, False)