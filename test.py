import tensorflow as tf

w1 = tf.Variable(1)
w2 = tf.Variable(2)

k = tf.add(w1, w2)
res = tf.multiply(k, k, name="d")

grads = tf.gradients(res, w1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    re = sess.run(grads)
    print(re)
