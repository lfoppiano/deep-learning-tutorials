import tensorflow as tf
from tensorflow.keras import layers



model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation=tf.relu))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))


# # A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
# layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
#
# # A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
# layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))
#
# # A linear layer with a kernel initialized to a random orthogonal matrix:
# layers.Dense(64, kernel_initializer='orthogonal')
#
# # A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))


model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

