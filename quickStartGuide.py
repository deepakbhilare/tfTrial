import tensorflow as tf
print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution is: {}".format(tf.executing_eagerly()))
print("Keras version: {}".format(tf.keras.__version__))

vdataset = learn.datasets.mnist.read_data_sets('MNIST-data', one_hot=True)
