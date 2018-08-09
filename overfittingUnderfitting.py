import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


NUM_WORDS = 10000
(train_data, train_label), (test_data, test_label) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)


def multihot_sequences(sequence, dimension):
    sequences = np.zeros((sequence.shape[0], dimension))
    for i, l in enumerate(sequence):
        sequences[i, l] = 1.0
    return sequences


train_seq = multihot_sequences(train_data, NUM_WORDS)

plt.figure()
plt.plot(train_seq[0])

baseline_model = keras.models.Sequential([
    tf.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    tf.layers.Dense(16, activation=tf.nn.relu),
    tf.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer=tf.train.AdamOptimizer(),
                       loss=keras.losses.binary_crossentropy,
                       metrics=[keras.metrics.binary_crossentropy, keras.metrics.binary_accuracy])

baseline_model.summary()

baseline_history = baseline_model.fit(train_seq,
                             train_label,
                             batch_size=512,
                             epochs=20,
                             validation_split=0.2,
                             verbose=2)

smaller_model = keras.models.Sequential([
    tf.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    tf.layers.Dense(4, activation=tf.nn.relu),
    tf.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer=tf.train.AdamOptimizer(),
                      loss=keras.losses.binary_crossentropy,
                      metrics=[keras.metrics.binary_crossentropy, keras.metrics.binary_accuracy])

smaller_model.summary()

smaller_history = smaller_model.fit(train_seq,
                                    train_label,
                                    batch_size=512,
                                    epochs=20,
                                    validation_split=0.2,
                                    verbose=2)

