import tensorflow as tf
import numpy as np

from tensorflow import keras
import matplotlib.pyplot as plt


# Function to translate imdb reviews from integers
def translate_from_int_to_string(sentence):
    return ' '.join([reverse_word_index[x] for x in sentence])


# Get IMDB data
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=10000)
# Get IMDB word index dictionary
word_index = keras.datasets.imdb.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# Create dictionary to get words from integer index
reverse_word_index = {v: k for k, v in word_index.items()}

# Preprocess data by padding the spaces and making inputs the same length
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        maxlen=256,
                                                        value=word_index['<PAD>'],
                                                        padding='post')

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       maxlen=256,
                                                       value=word_index['<PAD>'],
                                                       padding='post')

# Create Model
# 1. Layer: Embedding layer
vocab_size = 10000
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

print(model.summary())

# Compile model with Adam Optimizer, binary crossentropy loss function and accuracy metrics
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
                    #validation_data=(validation_data, validation_labels),
                    validation_split=0.4,
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



