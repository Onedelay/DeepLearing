from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

tmp = [[0 for col in range(784)] for row in range(10)]

fig = plt.figure()

for i in range(10):
    tmp[i] = mnist.train.images[i]
    tmp[i] = tmp[i].reshape((28,28))
    
    subplot = fig.add_subplot(2,5,i+1)
    subplot.imshow(tmp[i].reshape((28,28)), cmap=cm.gray)

plt.show()
print('???')
