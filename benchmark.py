from __future__ import print_function

import tensorflow as tf

import util
import network_structure as ns


def generate_accuracy(train_path, test_path, learning_rate, training_epochs, lower_bound, upper_bound,):

    print("=========BENCH_MARK===========")
    # learning_rate = 0.001
    # training_epochs = 10

    train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_path, test_path, read_next=False)

    net_stru = ns.NNStructure(train_set_X[0], learning_rate)

    # newgrads = tf.gradients(loss_op, weights["h1"])

    y = None
    train_acc = 0
    test_acc = 0

    with tf.Session() as sess:
        sess.run(net_stru.init)

        data_size = len(train_set_X)
        data_size =30

        # print("x:", train_set_X[:data_size])

        for epoch in range(training_epochs):
            _, c, predicted_y = sess.run([net_stru.train_op, net_stru.loss_op, net_stru.logits],
                                         feed_dict={net_stru.X: train_set_X[:data_size], net_stru.Y: train_set_Y[:data_size]})
        print("Optimization Finished!")

        train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
        test_y = sess.run(net_stru.logits, feed_dict={net_stru.X: test_set_X})

        train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
        test_acc = util.calculate_accuracy(test_y, test_set_Y, False)

        predicted = tf.cast(net_stru.logits > 0.5, dtype=tf.float32)
        util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X:x}), train_set_X[:data_size], train_set_Y[:data_size],
                                    lower_bound, upper_bound, -1)

    tf.reset_default_graph()
    return train_acc, test_acc
