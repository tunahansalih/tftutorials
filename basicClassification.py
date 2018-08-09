import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_label_names = np.array([class_names[x] for x in train_labels])
test_label_names = np.array([class_names[x] for x in test_labels])

# Make values between 0.0 and 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# Show 10 training image
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.xlabel(train_label_names[i])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

# Create sequential model
# 1. layer: Flatten the (28,28) images to (28*28,1) arrays
# 2. layer: Dense layer with relu(f(x) = max(0,x))activation function
# 3. layer: Dense layer with softmax (softmax is used to evaluate probabilities of belonging to a class)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax),
])

# To compile we need optimizer, loss function and metrics
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train labels
model.fit(train_images, train_labels, epochs=5)


test_loss, test_accuracy = model.evaluate(test_images, test_labels)

predictions = np.argmax(model.predict(test_images), axis=1)
print(predictions)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    lbl = predictions[i]
    if lbl == test_labels[i]:
        plt.xlabel(class_names[lbl], color='green')
    else:
        plt.xlabel('%(expected)s, %(predicted)s' % \
                   {'expected': class_names[test_labels[i]], 'predicted': class_names[predictions[i]]},
                   color='red')
    plt.imshow(test_images[i], cmap=plt.cm.binary)

plt.show()