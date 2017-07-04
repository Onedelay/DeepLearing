import tensorflow as tf

W = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


hypothesis = X * W + b

cost = tf.reduce_sum(tf.square(hypothesis - Y))

learning_rate = 0.01

gradient = tf.reduce_mean((W * X + b - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

gradient_b = tf.reduce_mean((W * X + b) -Y)
descent_b = b - learning_rate * gradient_b
update_b = b.assign(descent_b)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(10):
    cost_val, gradient_val, descent_val, descent_b_val, W_, _, _ = sess.run([cost, gradient, descent, descent_b, W, update, update_b], feed_dict={X: [1,2,3], Y:[1,2,3]})
    print(cost_val, gradient_val, descent_val, descent_b_val, W_)
