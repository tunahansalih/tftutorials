import tensorflow as tf
import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt


def translate_from_int_to_string(sentence):
    return ' '.join([reverse_word_index[x] for x in sentence])


(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)

word_index = keras.datasets.imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = {v: k for k, v in word_index.items()}

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        maxlen=256,
                                                        value=word_index['<PAD>'],
                                                        padding='post')

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       maxlen=256,
                                                       value=word_index['<PAD>'],
                                                       padding='post')

vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

validation_data = train_data[:10000]
train_data = train_data[10000:]

validation_labels = train_labels[:10000]
train_labels = train_labels[10000:]

history = model.fit(train_data,
                    train_labels,
                    batch_size=512,
                    epochs=40,
                    validation_data=(validation_data, validation_labels),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)

plt.figure()
i = 1
for k, v in history.history.items():
    plt.subplot(2, 2, i)
    plt.plot(range(1, len(v)+1), v)
    plt.ylim((0, max(max(v), 1)))
    plt.xlim((1, len(v)))
    plt.grid(True)

    plt.title(k)
    i += 1
plt.show()



