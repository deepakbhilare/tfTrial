from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf



g = tf.Graph()
with g.as_default():
    a = tf.constant(2.5, name = 'first_val')
    b = tf.constant(4.5, name = 'second_val')
    sum = a + b
    prod = a * b
print(g.get_tensor_by_name('first_val:0'))
print(g.get_operations())

