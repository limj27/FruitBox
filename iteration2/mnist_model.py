import numpy as np
import mnist
import tensorflow as tf

X_train = mnist.train_images
y_train = mnist.train_labels
X_test = mnist.test_images
y_test = mnist.test_labels

X_train = (X_train / 255) - 0.5
X_test = (X_test / 255) - 0.5

X_train = X_train.reshape((-1, 784))
X_test = X_test.reshape((-1, 784))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  X_train, # training data
  y_train, # training targets
  epochs=5,
  batch_size=32,
)