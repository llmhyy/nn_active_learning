def confirm_gradient_direction(sess, points, aggregated_network, gradients):
    delta = 10E-6

    points_add = []
    points_minus = []
    for k in range(len(points)):
        point = points[k]
        gradient = gradients[k]
        point_add = []
        point_minus = []
        for i in range(len(point)):
            value = point[i]
            g = gradient[0][i]
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
        gradient = gradients[i]
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
