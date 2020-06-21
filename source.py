import tensorflow as tf
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)

model = tf.keras.models.Sequential()
model.add(Flatten())
model.add(Dense(64, activation = tf.nn.relu))
model.add(Dense(64, activation = tf.nn.relu))
model.add(Dense(10, activation = tf.nn.softmax))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', #USE BINARY WHEN SWITCHING
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=3)

predictions = model.predict(x_test)
print(np.argmax(predictions[0]))

plt.imshow(x_test[0], cmap = plt.cm.binary)
plt.show()

print(y_test[0])