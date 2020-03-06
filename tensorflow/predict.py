import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0



model = tf.keras.models.load_model('model.h5')



# wrap the trained model to return a probablility
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)



#===== plot functions =====#

def plot_image(image):
	plt.axis('off')
	plt.imshow(image, cmap=plt.cm.binary)

def plot_verify(prediction, pred_label, true_label):
	plt.axis('on')
	plt.xticks(range(10))
	plt.yticks([])
	thisplot = plt.bar(range(10), prediction, color="#888888")
	plt.ylim([0, 1])
	thisplot[pred_label].set_color('red')
	thisplot[true_label].set_color('blue')



#===== incorrects =====#

pred_labels = np.argmax(predictions, axis=1)
incorrectIdx = [idx for idx in range(len(test_labels)) if pred_labels[idx] != test_labels[idx]]

incorrects = len(incorrectIdx)
print(f'number of incorrect predicitons = {incorrects}')



#===== unconfident =====#

probablility = 0.65
confidence = np.max(predictions, axis = 1)
unconfIdx = [idx for idx in range(len(confidence)) if confidence[idx] < probablility]

unconfidents = len(unconfIdx)
print(f'number of unconfident(<{probablility}) guesses = {unconfidents}')



col = 5
row = 6
rg0 = row * col
rg1 = unconfidents + rg0 - 1 // rg0
col *= 2
plt.figure(figsize=[8,6])
plt.axis('equal')

for i0 in range(rg0):
	idx = unconfIdx[i0]
	plt.subplot(row, col, i0*2 + 1)
	plot_image(test_images[idx])

	plt.subplot(row, col, i0*2 + 2)
	plot_verify(predictions[idx], pred_labels[idx], test_labels[idx])


plt.show()




#===== incorrects =====#

# pred_labels = np.argmax(predictions, axis=1)
# incorrectIdx = [idx for idx in range(len(test_labels)) if pred_labels[idx] != test_labels[idx]]

# incorrects = len(incorrectIdx)
# print(f'number of incorrect predicitons = {incorrects}')


#===== plot incorrects =====#

# col = 5
# row = 6
# rg0 = row * col
# rg1 = incorrects + rg0 - 1 // rg0
# col *= 2
# plt.figure(figsize=[8,6])
# plt.axis('equal')

# for i0 in range(rg0):
# 	idx = incorrectIdx[i0]
# 	plt.subplot(row, col, i0*2 + 1)
# 	plot_image(test_images[idx])

# 	plt.subplot(row, col, i0*2 + 2)
# 	plot_verify(predictions[idx], pred_labels[idx], test_labels[idx])


# plt.show()





# for i in range(rng0):
# 	idx = incorrectIdx[i]
# 	plt.subplot(row, col, i*2 + 1)
# 	plt.axis('off')
# 	plt.imshow(test_images[idx], cmap=plt.cm.binary)

# 	plt.subplot(row, col, i*2 + 2)
# 	plt.axis('on')
# 	plt.xticks(range(10))
# 	plt.yticks([])
# 	thisplot = plt.bar(range(10), predictions[idx], color="#888888")
# 	plt.ylim([0, 1])
# 	thisplot[pred_labels[idx]].set_color('red')
# 	thisplot[test_labels[idx]].set_color('blue')
# plt.show()







# print(len(train_labels), len(test_labels))

# n = 10
# plt.figure(figsize=[6.4,6.4])
# for i in range(n * n):
# 	plt.subplot(n, n, i+1)
# 	plt.axis('off')
# 	plt.imshow(test_images[i], cmap=plt.cm.binary)
#  	# plt.imsave(f'test_images/test_image_{i}.png', test_images[i], cmap=plt.cm.binary)
# plt.show()
