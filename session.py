import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Define a trainable variable
x_var = tf.Variable(0., name='x_result')
# Define an untrainable variable to hold the global step step_var = tf.Variable(0, trainable=False)
# Express loss in terms of the variable loss = x_var * x_var - 4.0 * x_var + 5.0
# Find variable value that minimizes loss
learn_rate = 0.1
num_epochs = 40
#optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=step_var)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

#init = tf.global_variables_initializer()

#saver = tf.train.Saver()
saver = tf.compat.v1.train.Saver({'v1': x_var})

summary_op = tf.summary.scalar('x', x_var)

sess = tf.compat.v1.Session()

g = tf.Graph()

with g.as_default():
    c = x_var
   # assert c.graph is g

#file_writer = tf.summary.FileWriter('log', graph=tf.get_default_graph())
#file_writer = tf.compat.v1.summary.FileWriter('/tmp/', sess.graph)

# Launch session
#with tf.Session() as sess:
 #   sess.run(init)
for epoch in range(num_epochs):
    _, step, result, summary = run([optimizer, step_var, x_var,
                                         summary_op])
print('Step %d: Computed result = %f' % (step, result))
# Print summary data file_writer.add_summary(summary, global_step=step) file_writer.flush()
# Store variable data
saver.save(sess, os.getcwd() + '/output')
print('Final x_var: %f' % sess.run(x_var))
