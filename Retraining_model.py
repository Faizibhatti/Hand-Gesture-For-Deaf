
from google.colab import auth
auth.authenticate_user()

from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
vcode = getpass.getpass()


import os
os.chdir("../asl-alphabet")


import numpy as np
import pandas as pd 
import os
import cv2

import tensorflow as tf
tf.set_random_seed(2)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import keras
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import regularizers


train_dir = '../asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_dir = '../asl-alphabet/asl_alphabet_test/asl_alphabet_test'

def load_data():
    images = []
    labels = []
    size = 64,64
    print("LOADING DATA FROM : ",end = "")
    for folder_index, folder in enumerate(os.listdir(train_dir)):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            labels.append(folder_index)
        
    images = np.array(images)
    images = images.astype('float32')/255.0
    
    labels = keras.utils.to_categorical(labels)   
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.1)
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test

x_train, x_test, y_train, y_test = load_data()

# Build the model

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense
from keras.models import Sequential
# tf.reset_default_graph()

model = Sequential()
# First convolutional layer accepts image input
model.add(Conv2D(filters=5, kernel_size=5, padding='same', activation='relu', 
                        input_shape=(64, 64, 3)))
# Add a max pooling layer
model.add(MaxPooling2D((4,4)))
# Add a convolutional layer
model.add(Conv2D(filters=15, kernel_size=(5,5), activation='relu', padding='same'))
# Add another max pooling layer
model.add(MaxPooling2D(pool_size=(4,4)))
# Flatten and feed to output layer
model.add(Flatten())
model.add(Dense(29, activation='softmax'))

# Summarize the model
model.summary()

# Compile the model
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
hist = model.fit(x_train, y_train, validation_split=0.2, epochs=10, batch_size=32)

# visualize the training and validation loss

hist = model.history
epochs = range(1, len(hist.history['loss']) + 1)

plt.subplots(figsize=(15,6))
plt.subplot(121)

plt.plot(epochs, hist.history['loss'], 'bo-')

plt.plot(epochs, hist.history['val_loss'], 'ro--')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(122)
plt.plot(epochs, hist.history['acc'], 'bo-')
plt.plot(epochs, hist.history['val_acc'], 'ro--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()


from keras.models import load_model

model.save('model.h5')