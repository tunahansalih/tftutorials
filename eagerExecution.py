import tensorflow as tf
import numpy as np

# Enable eager execution
tf.enable_eager_execution()


# Tensorflow operations on matrices
print(tf.add(2, 3))
print(tf.add([2, 3], [3, 5]))
print(tf.multiply([2, 3], [3, 5]))
x = tf.matmul([[2], [1]], [[3, 5]])
print(x.shape)
print(x.dtype)

# Numpy to tensorflow, tensorflow to numpy
ndarr = np.ones([3, 3])
tensor = tf.multiply(ndarr, 17)
print(tensor)
print(tensor.numpy())

# IS GPU available
print(tf.test.is_gpu_available())
print(x.device)

# Using Data API of Tensorflow
ds_tensor = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9])

# Create a CSV file
import tempfile

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 0
    Line 1
    Line 2
    Line 3""")

ds_file = tf.data.TextLineDataset(filename)
print(ds_tensor.map(tf.square))
print(ds_tensor.shuffle(2))
print(ds_tensor.batch(2))

for x in ds_tensor.batch(2):
    print(x)