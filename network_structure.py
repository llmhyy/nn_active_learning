import numpy as np
import tensorflow as tf


class NNStructure():

    def __init__(self, data_size, learning_rate):
        self.learning_rate = learning_rate

        # Network Parameters
        n_hidden_1 = 8  # 1st layer number of neurons
        n_hidden_2 = 8  # 2nd layer number of neurons
        n_input = len(data_size)
        n_classes = 1

        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0)) / np.sqrt(n_input/2),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0)) / np.sqrt(n_hidden_1/2),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes], mean=0)) / np.sqrt(n_hidden_2/2)
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        self.logits = self.multilayer_perceptron(self.X, self.weights, self.biases)
        self.probability = tf.nn.sigmoid(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))

        # self.logits = tf.nn.sigmoid(self.logits)
        # self.sig = - self.Y * tf.log(self.logits) - (1-self.Y) * tf.log(1-self.logits)
        # self.loss_op = tf.reduce_mean(self.sig)

        # self.A = tf   .square(self.logits - self.Y)
        # self.loss_op = tf.reduce_sum(tf.square(self.logits - self.Y))
        # self.loss_op = tf.reduce_mean(tf.square(self.logits - self.Y))
        # self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.logits)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Initializing the variables
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()

    # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        # x0 = tf.nn.batch_normalization(x, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        x0 = x
        self.layer_1 = tf.nn.relu(tf.add(tf.matmul(x0, weights['h1']), biases['b1']))

        self.layer_1 = tf.nn.batch_normalization(self.layer_1, mean=0.01, variance=1, offset=0, scale=1,
                                                 variance_epsilon=0.001)
        # layer1_out = tf.sigmoid(layer_1)

        # Hidden fully connected layer with 256 neurons
        self.layer_2 = tf.nn.relu(tf.add(tf.matmul(self.layer_1, weights['h2']), biases['b2']))

        self.layer_2 = tf.nn.batch_normalization(self.layer_2, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # layer2_out = tf.sigmoid(layer_2)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(self.layer_2, weights['out']) + biases['out']
        return out_layer


class AggregateNNStructure():
    def __init__(self, data_size, weights_dict_list, biases_dict_list):
        n_input = len(data_size)
        n_classes = 1

        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights_dict_list = weights_dict_list
        self.biases_dict_list = biases_dict_list

        # Construct model
        self.probability = self.construct_network()
        self.init = tf.global_variables_initializer()

    # Create model
    def construct_network(self):
        x = self.X
        aggregate_num = len(self.weights_dict_list)

        x0 = tf.nn.batch_normalization(x, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)

        sig_list = []
        for i in range(aggregate_num):
            weights_dict = self.weights_dict_list[i]
            bias_dict = self.biases_dict_list[i]
            layer_relu = tf.nn.relu(tf.add(tf.matmul(x0, weights_dict['h1']), bias_dict['b1']))
            layer_batch_norm = tf.nn.batch_normalization(layer_relu, mean=0.01, variance=1, offset=0, scale=1,
                                                         variance_epsilon=0.001)
            layer_sigmoid = tf.nn.sigmoid(tf.matmul(layer_batch_norm, weights_dict['out']) + bias_dict['out'])
            sig_list.append(layer_sigmoid)

        output = sig_list[0]
        for i in range(len(sig_list) - 1):
            output_tmp = sig_list[i + 1]
            output = tf.add(output, output_tmp)

        output = tf.divide(output, aggregate_num)
        return output

# class NNStructure_save():
#
#     def __init__(self, data_size, learning_rate):
#         self.learning_rate = learning_rate
#
#         # Network Parameters
#         n_hidden_1 = 1024 * 5  # 1st layer number of neurons
#         n_hidden_2 = 10  # 2nd layer number of neurons
#         n_input = len(data_size)  # MNIST data input (img shape: 28*28)
#         n_classes = 1  # MNIST total classes (0-9 digits)
#
#         # tf Graph input
#         self.X = tf.placeholder("float", [None, n_input])
#         self.Y = tf.placeholder("float", [None, n_classes])
#
#         # Store layers weight & bias
#         self.weights = {
#             'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0)),
#             'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0)),
#             'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0))
#         }
#         self.biases = {
#             'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#             'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#             'out': tf.Variable(tf.random_normal([n_classes]))
#         }
#
#         # Construct model
#         self.logits = self.multilayer_perceptron(
#             self.X, self.weights, self.biases)
#
#         # Define loss and optimizer
#         self.loss_op = tf.reduce_mean(
#             tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
#         self.optimizer = tf.train.AdamOptimizer(
#             learning_rate=learning_rate)
#         # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         # Initializing the variables
#         self.train_op = self.optimizer.minimize(self.loss_op)
#         # Initializing the variables
#         self.init = tf.global_variables_initializer()
#
#     # Create model
#
#     def multilayer_perceptron(self, x, weights, biases):
#         # Hidden fully connected layer with 256 neurons
#         x0 = tf.nn.batch_normalization(
#             x, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
#         x0 = x
#         layer_1 = tf.nn.relu(
#             tf.add(tf.matmul(x0, weights['h1']), biases['b1']))
#
#         layer_1 = tf.nn.batch_normalization(
#             layer_1, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
#         # layer1_out = tf.sigmoid(layer_1)
#
#         # Hidden fully connected layer with 256 neurons
#         # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
#
#         # layer_2 = tf.nn.batch_normalization(layer_2, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
#         # layer2_out = tf.sigmoid(layer_2)
#
#         # Output fully connected layer with a neuron for each class
#         out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#         return out_layer
