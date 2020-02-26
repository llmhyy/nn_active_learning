from __future__ import print_function

import tensorflow as tf

from main import util, network_structure as ns


def generate_accuracy(train_set_x, train_set_y, test_set_x, test_set_y, learning_rate, training_epochs, lower_bound,
                      upper_bound, model_folder, model_file):
    print("=========BENCH_MARK===========")
    # tf.reset_default_graph()

    net = ns.NNStructure(len(train_set_x[0]), learning_rate)
    # net = ns.NNStructure(learning_rate=learning_rate, shape=(1, 2))

    tf.summary.scalar("conf_loss", net.loss_op)
    tf.summary.scalar("learning_rate", net.learning_rate)
    write_op = tf.summary.merge_all()

    with tf.Session() as sess:

        sess.run(net.init)

        summary_writer = tf.summary.FileWriter("./log", graph=sess.graph)
        data_size = len(train_set_x)

        predicted = tf.cast(net.probability > 0.5, dtype=tf.float32)

        for epoch in range(training_epochs):
            _, loss, summary = sess.run(
                [net.train_op, net.loss_op, write_op],
                feed_dict={net.X: train_set_x[:data_size],
                           net.Y: train_set_y[:data_size]})
            # summary_writer.add_summary(summary, epoch)

            # if epoch % 200 == 0:
            #     util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
            #                                 train_set_x[:data_size], train_set_y[:data_size],
            #                                 lower_bound, upper_bound, epoch)

        util.plot_decision_boundary(lambda x: sess.run(predicted, feed_dict={net.X: x}),
                                    train_set_x[:data_size], train_set_y[:data_size],
                                    lower_bound, upper_bound, 0)
        util.save_model(sess, model_folder, model_file)
        # for op in tf.get_default_graph().get_operations():
        #     print(str(op.name))
        # net.print_parameters(sess)

        train_y = sess.run(net.probability, feed_dict={net.X: train_set_x})
        train_acc = util.calculate_accuracy(train_y, train_set_y, False)

        test_y = sess.run(net.probability, feed_dict={net.X: test_set_x})
        test_acc = util.calculate_accuracy(test_y, test_set_y, False)

        print("train:", train_acc, " test: ", test_acc)

    return train_acc, test_acc
