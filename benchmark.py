from __future__ import print_function

import tensorflow as tf

import network_structure as ns
import util


def generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate, training_epochs, lower_bound,
                      upper_bound):
    print("=========BENCH_MARK===========")

    net = ns.NNStructure(train_set_x[0], learning_rate)
    train_acc = 0
    test_acc = 0

    with tf.Session() as sess:
        sess.run(net.init)

        data_size = len(train_set_x)

        best_accuracy = 0
        loss_list = []
        predicted = tf.cast(net.probability > 0.5, dtype=tf.float32)

        for epoch in range(training_epochs):
            _, loss, = sess.run(
                [net.train_op, net.loss_op],
                feed_dict={net.X: train_set_x[:data_size],
                           net.Y: train_set_y[:data_size]})

            print("loss: ", loss, "temp: ")
            loss_list.append(loss)

        util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                                    train_set_x[:data_size], train_set_y[:data_size],
                                    lower_bound, upper_bound, 0)
        # saver = tf.train.Saver()
        # saver.restore(sess, "./models/benchmark.ckpt")

        train_y = sess.run(net.probability, feed_dict={net.X: train_set_x})
        train_acc = util.calculate_accuracy(train_y, train_set_y, False)

        test_y = sess.run(net.probability, feed_dict={net.X: test_set_x})
        test_acc = util.calculate_accuracy(test_y, test_set_y, False)

        print("train:", train_acc, " test: ", test_acc)

    return train_acc, test_acc
