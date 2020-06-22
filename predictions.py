import tensorflow as tf
import numpy as np
import os
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import array_to_img, img_to_array, load_img

model = load_model("my_model")

directory = os.fsencode('data/test/dogs')
zero_counter = 0

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = load_img('data/test/dogs/' + filename)
    img = img.resize((150, 150))
    x = img_to_array(img) # This is a Numpy array with shape (150, 150, 3)
    x = x.reshape((1,) + x.shape) # This is a Numpy array with shape (1, 150, 150, 3)

    prediction = model.predict(x)
    if round(prediction[0][0]) == 0: zero_counter += 1
print(zero_counter)
    



# img = load_img('data/test/dogs/dog.1004.jpg')  # this is a PIL image
# img = img.resize((150, 150))
# x = img_to_array(img)  # this is a Numpy array with shape (150, 150, 3)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 150, 150, 3)
# print(x.shape)

# prediction = model.predict(x)
# print(round(prediction[0][0]))