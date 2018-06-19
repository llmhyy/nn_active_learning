from __future__ import print_function
import tensorflow as tf
import csv
import numpy as np
import random
import math
# Parameters
learning_rate = 3
training_epochs = 3
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
		spamreader = csv.reader(csvfile)
		writer = csv.writer(file)
		i=0
		for row in spamreader:
			if(i>=70):
				break
			i+=1
			writer.writerow(row)

	file.close()


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

# ten times training
with tf.Session() as sess:
	sess.run(init)
	##data processing


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

	with open('train_next.csv', 'rt') as csvfile:
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



	h1 = sess.run(weights["h1"])
	out = sess.run(weights["out"])

	print("h1", h1)
	print("out", out)

	g = sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})
	##print(g)

	for i in range(10):
		for epoch in range(training_epochs):
			avg_cost = 0.
			total_batch = 700
 			_, c = sess.run([train_op, loss_op], feed_dict={X: train_set_X,
                                                        Y: train_set_Y})

			g = sess.run(newgrads, feed_dict={X: train_set_X, Y: train_set_Y})
			print(g)
			print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(c))

		print(str(i)+"Turn Optimization Finished!")
		train_y = sess.run(logits, feed_dict={X: train_set_X})
		test_y = sess.run(logits, feed_dict={X: test_set_X})

		##print(len(train_y))
		##print(len(train_set_Y))
		calculateAccuracy(train_y, train_set_Y, False)
		calculateAccuracy(test_y, test_set_Y, False)
		new_train_set_X=[]
		new_train_set_Y=[]
		for j in range(len(train_set_X)):
			tmpX1=train_set_X[j][0]+g[0][j][0]*learning_rate
			tmpX2=train_set_X[j][1]+g[0][j][1]*learning_rate

			if (g[0][j][0]>0.03):
				new_train_set_X.append([tmpX1,tmpX2])

				if((tmpX1-12.5)*(tmpX1-12.5)+tmpX2*tmpX2<100 or (tmpX1+12.5)*(tmpX1+12.5)+tmpX2*tmpX2<100):
					new_train_set_Y.append([0])
				else:
					new_train_set_Y.append([1])


		train_set_X=train_set_X+new_train_set_X
		train_set_Y=train_set_Y+new_train_set_Y

		print(len(train_set_X))
		print(len(train_set_Y))





