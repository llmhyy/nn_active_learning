import tensorflow as tf


class NNStructure():

    def __init__(self, data_size, learning_rate):
        self.learning_rate = learning_rate

        # Network Parameters
        n_hidden_1 = 1280  # 1st layer number of neurons
        n_hidden_2 = 10  # 2nd layer number of neurons
        n_input = len(data_size) # MNIST data input (img shape: 28*28)
        n_classes = 1  # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], mean=0)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], mean=0)),
            'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes], mean=0))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Construct model
        self.logits = self.multilayer_perceptron(self.X, self.weights, self.biases)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Initializing the variables
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()


    # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        x0 = tf.nn.batch_normalization(x, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # x0 = x
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x0, weights['h1']), biases['b1']))

        layer_1 = tf.nn.batch_normalization(layer_1, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # layer1_out = tf.sigmoid(layer_1)

        # Hidden fully connected layer with 256 neurons
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        # layer_2 = tf.nn.batch_normalization(layer_2, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # layer2_out = tf.sigmoid(layer_2)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer



class NNStructureFixedVar():
    def __init__(self, data_size, learning_rate,weights,biases):
        self.learning_rate = learning_rate

        # Network Parameters

        n_input = len(data_size) # MNIST data input (img shape: 28*28)
        n_classes = 1  # MNIST total classes (0-9 digits)

        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        self.weights = weights
        self.biases = biases

        # Construct model
        self.logits = self.multilayer_perceptron(self.X, self.weights, self.biases)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # Initializing the variables
        self.train_op = self.optimizer.minimize(self.loss_op)
        # Initializing the variables
        self.init = tf.global_variables_initializer()


    # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden fully connected layer with 256 neurons
        x0 = tf.nn.batch_normalization(x, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # x0 = x
        layer_1 = tf.nn.relu(tf.add(tf.matmul(x0, weights['h1']), biases['b1']))

        layer_1 = tf.nn.batch_normalization(layer_1, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # layer1_out = tf.sigmoid(layer_1)

        # Hidden fully connected layer with 256 neurons
        # layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))

        # layer_2 = tf.nn.batch_normalization(layer_2, mean=0.01, variance=1, offset=0, scale=1, variance_epsilon=0.001)
        # layer2_out = tf.sigmoid(layer_2)

        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer