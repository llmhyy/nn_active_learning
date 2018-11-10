import numpy as np
import math
import util

def confirm_gradient_direction(sess, points, aggregated_network, gradients):
    delta = 10E-6

    points_add = []
    points_minus = []
    for k in range(len(points)):
        point = points[k]
        gradient = gradients[0][k]
        point_add = []
        point_minus = []
        for i in range(len(point)):
            value = point[i]
            g = gradient[i]
            value_add = value + delta * g
            value_minus = value - delta * g

            point_add.append(value_add)
            point_minus.append(value_minus)

        points_add.append(point_add)
        points_minus.append(point_minus)

    ys = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: points})
    ys_add = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: points_add})
    ys_minus = sess.run(aggregated_network.probability, feed_dict={aggregated_network.X: points_minus})

    returned_gradients = []
    for i in range(len(ys)):
        y = ys[i]
        y_add = ys_add[i]
        y_minus = ys_minus[i]
        gradient = gradients[0][i]
        if y < 0.5:
            if y_add > y_minus:
                returned_gradients.append(gradient)
            else:
                returned_gradients.append(-gradient)
        else:
            if y_add > y_minus:
                returned_gradients.append(-gradient)
            else:
                returned_gradients.append(gradient)

    return returned_gradients


def calculate_opposite_vector(vector):
    dimension_length = len(vector)
    if dimension_length == 1:
        return [0]

    random_vector = np.random.randn(dimension_length)
    vector_size = util.calculate_vector_size(vector)
    if vector_size == 0:
        return random_vector.tolist()

    vector = np.array(vector)
    while True:
        value = np.dot(random_vector, vector)
        random_vector_size = util.calculate_vector_size(random_vector)
        cosine = value/(random_vector_size*vector_size)

        if -math.pi/2 < cosine <= 0:
            return random_vector.tolist()
        else:
            random_vector = np.random.randn(dimension_length)

    return random_vector.tolist()

