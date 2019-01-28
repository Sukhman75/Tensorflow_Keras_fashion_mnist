
import tensorflow as tf
from tensorflow import keras 

import numpy as np
import matplotlib.pyplot as plt
#Import the Fashion MNIST dataset from keras
fashion_data = keras.datasets.fashion_mnist
#Spliting the data in train a
(train_image, train_labels), (test_image, test_labels) = fashion_data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

