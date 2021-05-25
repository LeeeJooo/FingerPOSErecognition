from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D
from keras.models import load_model
import os
import cv2
import numpy as np

finger_path = 'fingers'
categories = ['zero', 'one', 'two', 'three', 'four', 'five']
num_class = len(categories)
X = []
Y = []

for idx, category in enumerate(categories):
    label = [0 for i in range(num_class)]
    label[idx] = 1
    image_dir = finger_path + '/' + category + '/'
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir + filename)
            X.append(img/128)
            Y.append(label)

Xtr = np.array(X)
Ytr = np.array(Y)
X_train, Y_train = Xtr, Ytr

model = Sequential()
model.add(Conv2D(16, 3, 3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(20, 3, 3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(6, activation='softmax'))
print(model.summary())