#!/usr/bin/env python
import tensorflow as tf
import sys
import matplotlib.pyplot as plt

# Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares

# steps = [0.005, 0.001, 0.01, 0.02]
# for step in steps:
#     # optimizer
#     optimizer = tf.train.GradientDescentOptimizer(step)
#     train = optimizer.minimize(loss)
#
#     # training data
#     x_train = [1, 2, 3, 4]
#     y_train = [0, -1, -2, -3]
#     # training loop
#     init = tf.global_variables_initializer()
#     sess = tf.Session()
#     sess.run(init) # reset values to wrong
#
#     curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
#     print("%s,%s"%(0, curr_loss))
#
#     x_list = [0]
#     y_list = [curr_loss]
#
#     for i in range(1000):
#         sess.run(train, {x: x_train, y: y_train})
#         curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
#         print("%s,%s"%(i, curr_loss))
#         x_list.append(i)
#         y_list.append(curr_loss)
#
#     plt.plot(x_list,y_list,label=step)
#     plt.legend()

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
for optimizer in optimizers:
    print(optimizer.get_name())
    train = optimizer.minimize(loss)

    # training data
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # training loop
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init) # reset values to wrong

    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("%s,%s"%(0, curr_loss))

    x_list = [0]
    y_list = [curr_loss]

    for i in range(1000):
        sess.run(train, {x: x_train, y: y_train})
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
        print("%s,%s"%(i, curr_loss))
        x_list.append(i)
        y_list.append(curr_loss)

    plt.plot(x_list,y_list,label=optimizer.get_name())
    plt.legend()

plt.ylabel('loss')
plt.xlabel('iteration')
plt.show()



