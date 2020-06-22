import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Applying random transformations to images to not receive same image twice
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


## Define the model
# We used sigmoid since it is best for binary classification
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

print('Model Summary \n')
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

batch_size = 16

# Modifications to images used for training
train_data_generator = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Modifications to images used for testing (only scaled)
test_data_generator = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(
    'data/train', # Where the training images come from
    target_size=(150, 150), # Images resized to this size
    batch_size=batch_size,
    class_mode='binary' # Since we have 2 classes
)

test_generator = test_data_generator.flow_from_directory(
    'data/test', # Where the testing images come from
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch= 2000 // batch_size, # Since there are 1000 images of cats/1000 images of dogs
    epochs = 15,
    validation_data=test_generator,
    validation_steps= 800 // batch_size # Since there are 400 images of cats/400 images of dogs
)

model.save('my_model')

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()