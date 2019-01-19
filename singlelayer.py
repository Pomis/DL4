#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt
import time

start = time.time()

# read data from file
data_input = read_inputs.load_data_mnist('MNIST_data/mnist.pkl.gz')
# FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]
# print ( N.shape(data[0][0])[0] )
# print ( N.shape(data[0][1])[0] )

# data layout changes since output should an array of 10 with probabilities
real_output = N.zeros((N.shape(data[0][1])[0], 10), dtype=N.float)
for i in range(N.shape(data[0][1])[0]):
    real_output[i][data[0][1][i]] = 1.0

# data layout changes since output should an array of 10 with probabilities
real_check = N.zeros((N.shape(data[2][1])[0], 10), dtype=N.float)
for i in range(N.shape(data[2][1])[0]):
    real_check[i][data[2][1][i]] = 1.0

# set up the computation. Definition of the variables.
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

optimizers = [tf.train.GradientDescentOptimizer(0.01),
              tf.train.AdadeltaOptimizer(0.01),
              tf.train.AdagradOptimizer(0.01),
              tf.train.MomentumOptimizer(0.01, 0.9),
              tf.train.AdamOptimizer(0.01),
              tf.train.FtrlOptimizer(0.01),
              tf.train.ProximalGradientDescentOptimizer(0.01),
              tf.train.ProximalAdagradOptimizer(0.01),
              tf.train.RMSPropOptimizer(0.01)
              ]

# learning_rates = [0.005, 0.001, 0.002, 0.0001, 0.00001]
# learning_rates = [0.9, 0.5, 0.1, 0.05, 0.01, 0.001, 0.0001]
for optimizer in optimizers:
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # TRAINING PHASE
    print("TRAINING")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    loss = []
    iterations = []
    for i in range(500):
        batch_xs = data[0][0][100 * i:100 * i + 100]
        batch_ys = real_output[100 * i:100 * i + 100]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        loss.append(1 - sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))
        iterations.append(i)
        print(loss[i])

    plt.legend()

    # CHECKING THE ERROR
    print("ERROR CHECK")

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    plt.plot(iterations, loss, label=("Solver = " + optimizer.get_name() + ". Acc = " + str(sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check}))))


plt.show()
end = time.time()
totalTime = end - start
f = open(f'singlelayer_time{totalTime}.txt', 'w')
f.write(f'this is the  time {totalTime}')
f.close()