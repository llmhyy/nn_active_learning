from __future__ import print_function

import tensorflow as tf
from matplotlib import pyplot as plt

import network_structure as ns
import util


def generate_accuracy(train_path, test_path, learning_rate, training_epochs, lower_bound, upper_bound):
    print("=========BENCH_MARK===========")

    # learning_rate = 0.00001
    # training_epochs = 200

    train_set_X, train_set_Y, test_set_X, test_set_Y = util.preprocess(train_path, test_path, read_next=False)
    net_stru = ns.NNStructure(train_set_X[0], learning_rate)
    saver = tf.train.Saver()

    # newgrads = tf.gradients(loss_op, weights["h1"])

    y = None
    train_acc = 0
    test_acc = 0

    with tf.Session() as sess:
        sess.run(net_stru.init)

        data_size = len(train_set_X)
        # data_size = 100

        # print("x:", train_set_X[:data_size])

        best_accuracy = 0
        loss_list = []
        predicted = tf.cast(net_stru.probability > 0.5, dtype=tf.float32)
        # gra = tf.gradients(net_stru.loss_op, net_stru.weights["h1"])
        # gra2 = tf.gradients(net_stru.loss_op, net_stru.weights["out"])
        # gra3 = tf.gradients(net_stru.loss_op, net_stru.logits)
        # gra4 = tf.gradients(net_stru.logits, net_stru.weights["out"])
        # gra5 = tf.gradients(net_stru.logits, net_stru.layer_1)
        for epoch in range(training_epochs):
            _, loss, = sess.run(
                [net_stru.train_op, net_stru.loss_op],
                feed_dict={net_stru.X: train_set_X[:data_size],
                           net_stru.Y: train_set_Y[:data_size]})
            # _, loss, l, con_gra, con_gra2, con_gra3, con_gra4, con_gra5, A, layer1 = sess.run(
            #     [net_stru.train_op, net_stru.loss_op, net_stru.logits, gra, gra2,
            #      gra3, gra4, gra5, net_stru.A, net_stru.layer_1],
            #     feed_dict={net_stru.X: train_set_X[:data_size],
            #                net_stru.Y: train_set_Y[:data_size]})

            print("loss: ", loss, "temp: ")
            # print(con_gra)
            # print(train_set_Y[:3])
            # print(l[:3])
            # print(con_gra3[0][:3])
            # print(con_gra2[0])
            # print(con_gra4[0])
            loss_list.append(loss)
            if epoch % 10 == 0:
                util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net_stru.X: x}),
                                            train_set_X[:data_size], train_set_Y[:data_size],
                                            lower_bound, upper_bound, epoch)

            # train_y = sess.run(net_stru.logits, feed_dict={net_stru.X: train_set_X})
            # train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
            # if train_acc > best_accuracy:
            #     best_accuracy = train_acc
            #     print("best_accuracy ", best_accuracy)
            #     saver.save(sess, './models/benchmark.ckpt')

        print("Optimization Finished!")
        plt.clf()
        plt.plot(loss_list)
        file_name = 'trend.png'
        plt.savefig(file_name)
        # saver.restore(sess, "./models/benchmark.ckpt")

        train_y = sess.run(net_stru.probability, feed_dict={net_stru.X: train_set_X})
        test_y = sess.run(net_stru.probability, feed_dict={net_stru.X: test_set_X})

        train_acc = util.calculate_accuracy(train_y, train_set_Y, False)
        test_acc = util.calculate_accuracy(test_y, test_set_Y, False)

        print("train:", train_acc, " test: ", test_acc)

    return train_acc, test_acc
