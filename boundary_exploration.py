import tensorflow as tf

from prj_test import formula
import network_structure as ns
import util


def boundary_explore(train_set_X, train_set_Y, label, iteration, formu, lower_bound, upper_bound, learning_rate,
                     training_epochs):
    while True:
        point_added_number = len(train_set_X)
        assumption_points_label = 1 - label
        assumption_points_X, assumption_points_Y = append_assumption_points(assumption_points_label, point_added_number,
                                                                            formu, lower_bound, upper_bound)
        train_set_X = train_set_X + assumption_points_X
        train_set_Y = train_set_Y + assumption_points_Y
        net_stru = ns.NNStructure(train_set_X[0], learning_rate)
        with tf.Session() as sess:
            sess.run(net_stru.init)
            for epoch in range(training_epochs):
                _, c = sess.run([net_stru.train_op, net_stru.loss_op],
                                feed_dict={net_stru.X: train_set_X, net_stru.Y: train_set_Y})

        ##TODO add point near boundary


def append_assumption_points(assumption_points_label, point_added_number, formu, lower_bound, upper_bound):
    category = formu.get_category()
    if (category == formula.POLYNOMIAL):
        newPointsX, newPointsY = util.generate_polynomial_points(formu, point_added_number, lower_bound, upper_bound)
        for i in newPointsY:
            i[0] = assumption_points_label
        return newPointsX, newPointsY
    elif (category == formula.POLYHEDRON):
        newPointsX, newPointsY = util.generate_polyhedron_points(formu, point_added_number, lower_bound, upper_bound)
        for i in newPointsY:
            i[0] = assumption_points_label
        return newPointsX, newPointsY
