import tensorflow as tf

filename_queue = tf.train.string_input_producer(['HR_comma_sep.csv'], shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.float32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.int32)]

col0, col1, col2, col3, col4, col5, col6, col7, col8, col9 = tf.decode_csv(value, record_defaults=record_defaults)

assert col0.dtype == tf.float32
assert col1.dtype == tf.float32
assert col2.dtype == tf.int32
assert col3.dtype == tf.int32
assert col4.dtype == tf.int32
assert col5.dtype == tf.int32
assert col6.dtype == tf.int32
assert col7.dtype == tf.int32
assert col8.dtype == tf.int32
assert col9.dtype == tf.int32


train_x_batch, train_y_batch = tf.train.batch([col0, col1, col2, col3, col4, col5, col6, col7, col8], col9], batch_size=100)

X = tf.placeholder(tf.float32, shape=[None, 10])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([10,1], name='weight'))
b = tf.Variable(tf.random_normal([1], name='bias'))

hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean(tf.square(hypothesis = Y))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-5).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

print("test - ", sess.run(hypothesis, feed_dict={X: [[0.55, 0.43, 3,140,2,0,1,1,'support']]}))
