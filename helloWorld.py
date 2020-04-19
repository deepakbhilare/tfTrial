from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

print(tf.version)
print(tf._)
# Create tensor
t1 = tf.constant([[1.5, 2.5], [3.5,1.5], [2.5, 3.5]])
print(t1)

t2 = tf.constant([['b', 'b'], ['b', 'b']])
print(t2)

t3 = tf.constant([4, 2, 4], tf.int16, [3], 'Const')
print(t3)

zero_tensor = tf.zeros([3])
print(zero_tensor)

one_tensor = tf.ones([4,4])
print(one_tensor)

fill_tensor = tf.fill([1,2,3], 81.0)
print(fill_tensor)

lin_tensor = tf.linspace(5., 9., 5)
print(lin_tensor)

mat = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
slice_mat = tf.slice(mat, [0, 1], [2, 2])
print(slice_mat)

t1 = tf.constant([1, 2])
t2 = tf.constant([3, 4])
t3 = tf.constant([5, 6])
t4 = tf.stack([t1, t2, t3])
print(t4)

m1 = tf.constant([[1, 2], [3, 4]])
m2 = tf.constant([[5, 6], [7, 8]])
e1 = tf.einsum('ij->ji', m1)
print(e1)
e2 = tf.einsum('ij, jk->ik', m1, m2)
print(e2)
