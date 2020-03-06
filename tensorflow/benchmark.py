import tensorflow as tf
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0



# make the model of the network #
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(28, 28)),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10)
])



# comppile the model #
# - needs a few settings
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



# train the network #
# - feed the model training data
# - adjusts the model parameters to minimize the loss
model.fit(train_images, train_labels, epochs=5)



# evaluate accuracy
# - checks the models performance
model.evaluate(test_images, test_labels, verbose=2)
