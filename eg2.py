import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds

# tfds.disable_progress_bar()

print(tf.version)

embedding_layer = layers.Embedding(1000, 5)

tf.executing_eagerly()

result = embedding_layer(tf.constant([1, 2, 3]))
result.numpy()

result = embedding_layer(tf.constant([[0, 1, 2], [3, 4, 5]]))

#tfds.load()

(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)


print(info)
print(train_data)
print(test_data)
