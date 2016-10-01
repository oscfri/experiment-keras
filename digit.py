import numpy as np

import pandas as pd

from keras.models import Sequential
from keras.layers import (Dense, Convolution2D,
                          Flatten, Activation,
                          MaxPooling2D, BatchNormalization)

def to_vector(y):
    Y = np.zeros(10)
    Y[y] = 1
    return Y


train_file = "./kaggle/digit-recognizer/train.csv"
test_file = "./kaggle/digit-recognizer/test.csv"

image_width = 28
image_height = 28
image_size = image_width * image_height

pixel_header_names = ['pixel%d' % i for i in range(image_size)]
train_header_names = ["label"] + pixel_header_names
test_header_names = pixel_header_names

train_data = pd.read_csv(train_file, header=0, names=train_header_names,
                         dtype='int')
test_data = pd.read_csv(test_file, header=0, names=test_header_names,
                        dtype='int')

y_train = np.vstack(map(to_vector, train_data.pop('label')))
X_train = train_data.as_matrix().reshape((train_data.shape[0], 28, 28, 1))
X_test = test_data.as_matrix().reshape((test_data.shape[0], 28, 28, 1))

model = Sequential()
model.add(Convolution2D(20, 3, 3, input_shape=(image_width, image_height, 1), border_mode="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Convolution2D(10, 3, 3, border_mode="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Convolution2D(5, 3, 3, border_mode="same"))
model.add(Convolution2D(5, 3, 3, border_mode="same"))
model.add(Convolution2D(5, 3, 3, border_mode="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(optimizer='sgd', loss='categorical_crossentropy')

model.fit(X_train, y_train, nb_epoch=10, batch_size=32)

prediction = model.predict(X_test, batch_size=32).argmax(axis=1)

prediction_dataframe = pd.DataFrame(data=prediction)
prediction_dataframe.index += 1

prediction_dataframe.to_csv("out.csv", header=['Label'], index=True, index_label='ImageId')
