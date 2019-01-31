
import tensorflow as tf
from tensorflow import keras 

import numpy as np
import matplotlib.pyplot as plt
#Import the Fashion MNIST dataset from keras
fashion_data = keras.datasets.fashion_mnist
#Spliting the data in train a
(train_image, train_labels), (test_image, test_labels) = fashion_data.load_data()
# As the class in not defined in the Data set, we have to do it by storing a list of class name in a Object(class_name).
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# To explore the data & to look deep into data

# print(train_image.shape)
# print(train_labels)

# print(test_image.shape)

# Next step is to Preprocess the Data
""" It will deiplay an example image and the pixel value from the test Data"""

# plt.figure()
# plt.imshow(train_image[0])
# plt.xlabel(train_labels[0])
# plt.colorbar()
# plt.show()

#Now we will sacle the value from 0 to 255 to 0 to 1, by dividing the training and test images by 255.

train_image = train_image/255.0
test_image = test_image/255.0
#	to display first 25 images from the data
# plt.figure(figsize=(10,10))

# for i in range(25):
# 	plt.subplot(5,5,i+1)
# 	plt.xticks([])
# 	plt.yticks([])
# 	plt.grid(False)
# 	plt.imshow(train_image[i], cmap=plt.cm.binary)
# 	plt.xlabel(class_names[train_labels[i]])
# plt.show()

#Building the Model

#For Layering 

model = keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
]) 
#BEFORE TRAINING AND TESTING COMPILE THE MODEL
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#TRAIN THE MODEL
""" Here I used epoch value = 5 or you can compare different values,
    but don't put too much value it will overfit the model """
model.fit(train_image, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_image, test_labels)

print('Test accuracy:', test_acc)
print('Test Loss:', test_loss)

#MAKE PREDICTIONS

predictions = model.predict(test_image)

print(predictions[0], '\n', np.argmax(predictions[0]))
print(test_labels[0])
# DO A PROPER PLOT OF PREDICTIONS

plt.figure(figsize=(15,15))
for i in range(20):
	plt.subplot(5,4,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	predict_max = 100*np.max(predictions[i])
	real_class = class_names[test_labels[i]]
	predict_class = class_names[np.argmax(predictions[i])] 
	plt.imshow(test_image[i], cmap=plt.cm.binary)
	if test_labels[i] == np.argmax(predictions[i]):
		color = 'blue'
	else:
		color = 'red'
	
	
	plt.xlabel("{} {:2.0f}% ({})".format(predict_class, predict_max, real_class), color = color)
plt.show()



