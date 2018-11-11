import tensorflow as tf


class NNStructure():

    def __init__(self, input_dimension, learning_rate):
        self.learning_rate = learning_rate
        self.w_init = tf.contrib.layers.variance_scaling_initializer(uniform=False, factor=2.0, mode='FAN_IN',
                                                                     dtype=tf.float32)
        # Network Parameters
        n_hidden_1 = 8  # 1st layer number of neurons
        n_hidden_2 = 4  # 2nd layer number of neurons
        n_input = input_dimension
        n_classes = 1

        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input], name="X")
        self.Y = tf.placeholder("float", [None, n_classes], name="Y")

        # Store layers weight & bias
        # with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
        self.weights = {
            'h1': tf.get_variable(name="h1", shape=[n_input, n_hidden_1], initializer=self.w_init),
            'h2': tf.get_variable(name="h2", shape=[n_hidden_1, n_hidden_2], initializer=self.w_init),
            'hout': tf.get_variable(name="hout", shape=[n_hidden_2, n_classes], initializer=self.w_init)
        }

        self.biases = {
            'b1': tf.get_variable(name="b1", shape=[n_hidden_1], initializer=self.w_init),
            'b2': tf.get_variable(name="b2", shape=[n_hidden_2], initializer=self.w_init),
            'bout': tf.get_variable(name="bout", shape=[n_classes], initializer=self.w_init)
            # 'b1': tf.Variable(tf.random_normal([n_hidden_1]), name="b1"),
            # 'b2': tf.Variable(tf.random_normal([n_hidden_2]), name="b2"),
            # 'bout': tf.Variable(tf.random_normal([n_classes]), name="bout")
        }

        self.logits = self.multilayer_perceptron(self.X, self.weights, self.biases)
        self.probability = tf.nn.sigmoid(self.logits)
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.init = tf.global_variables_initializer()

    def restore_parameter(self):
        graph = tf.get_default_graph()
        self.X = graph.get_tensor_by_name("X:0")
        self.Y = graph.get_tensor_by_name("Y:0")
        for key in self.weights:
            self.weights[key] = graph.get_tensor_by_name(key+":0")
        for key in self.biases:
            self.biases[key] = graph.get_tensor_by_name(key+":0")
        # self.X = tf.get_collection("X")[0]
        # self.Y = tf.get_collection("Y")[0]
        # for key in self.weights:
        #     self.weights[key] = tf.get_collection(key)[0]
        # for key in self.biases:
        #     self.biases[key] = tf.get_collection(key)[0]

        self.logits = self.multilayer_perceptron(self.X, self.weights, self.biases)
        self.probability = tf.nn.sigmoid(self.logits)
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

    def print_parameters(self, sess):
        for key in self.weights:
            print(key, self.weights[key])
            print(sess.run(self.weights[key]))
        for key in self.biases:
            print(key, self.biases[key])
            print(sess.run(self.biases[key]))

    def save_parameters(self):
        tf.add_to_collection("X", self.X)
        tf.add_to_collection("Y", self.Y)
        for key in self.weights:
            tf.add_to_collection(key, self.weights[key])
        for key in self.biases:
            tf.add_to_collection(key, self.biases[key])

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

        self.layer_2 = tf.nn.batch_normalization(self.layer_2, mean=0.01, variance=1, offset=0, scale=1,
                                                 variance_epsilon=0.001)
        # layer2_out = tf.sigmoid(layer_2)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(self.layer_2, weights['hout']) + biases['bout']
        return out_layer


class AggregateNNStructure():
    def __init__(self, input_dimension, weights_dict_list, biases_dict_list):
        n_input = input_dimension
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
            layer_relu1 = tf.nn.relu(tf.add(tf.matmul(x0, weights_dict['h1']), bias_dict['b1']))
            layer_batch_norm1 = tf.nn.batch_normalization(layer_relu1, mean=0.01, variance=1, offset=0, scale=1,
                                                          variance_epsilon=0.001)
            layer_relu2 = tf.nn.relu(tf.add(tf.matmul(layer_batch_norm1, weights_dict['h2']), bias_dict['b2']))
            layer_batch_norm2 = tf.nn.batch_normalization(layer_relu2, mean=0.01, variance=1, offset=0, scale=1,
                                                          variance_epsilon=0.001)

            layer_sigmoid = tf.nn.sigmoid(tf.matmul(layer_batch_norm2, weights_dict['hout']) + bias_dict['bout'])
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
