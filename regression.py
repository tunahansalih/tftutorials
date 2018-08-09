import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

boston_housing = keras.datasets.boston_housing

(train_data, train_label), (test_data, test_label) = boston_housing.load_data()

np.random.shuffle(train_data)
np.random.shuffle(train_label)

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

def build_model():
    model = keras.Sequential()
    model.add(keras.layers.Dense(64,
                                 activation=tf.nn.relu,
                                 input_shape=(train_data.shape[1],)))
    model.add(keras.layers.Dense(64,
                                 activation=tf.nn.relu))
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse',
                  metrics=[keras.metrics.mae])

    return model


model = build_model()
model.summary()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            print('', end='\n')
        print(str(epoch), end=' ')

#
# history = model.fit(train_data,
#                     train_label,
#                     epochs=500,
#                     validation_split=0.2,
#                     verbose=0,
#                     callbacks=[PrintDot()])
#
#
# plt.figure()
# i = 1
# for k, v in history.history.items():
#     plt.subplot(2, 2, i)
#     plt.plot(range(1, len(v)+1), v)
#     plt.grid(True)
#
#     plt.title(k)
#     i += 1
# plt.show()

history = model.fit(train_data,
                    train_label,
                    epochs=500,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[keras.callbacks.EarlyStopping('val_loss', patience=20),
                               PrintDot()])

plt.figure()
i = 1
for k, v in history.history.items():
    #plt.subplot(2, 2, i)
    plt.plot(range(1, len(v)+1), v, label=k)
    plt.grid(True)
    plt.legend()
    i += 1
plt.show()

print(model.evaluate(test_data, test_label, verbose=0))

print(model.predict(test_data).flatten())