import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Activation,
                          Embedding, Flatten, Convolution1D,
                          Merge, MaxPooling1D)
from keras.datasets import imdb

maxlen_left = 500
maxlen_right = 100
nb_words = 5000

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(nb_words=nb_words)
X_train_left = sequence.pad_sequences(X_train, maxlen=maxlen_left)
X_test_left = sequence.pad_sequences(X_test, maxlen=maxlen_left)
X_train_right = sequence.pad_sequences(X_train, maxlen=maxlen_right)
X_test_right = sequence.pad_sequences(X_test, maxlen=maxlen_right)

left_branch = Sequential()
right_branch = Sequential()
left_branch.add(Embedding(nb_words, 50, input_length=maxlen_left))
left_branch.add(Convolution1D(50, 3))
left_branch.add(MaxPooling1D(3))
left_branch.add(Flatten())
left_branch.add(Dense(10))
left_branch.add(Activation("sigmoid"))

right_branch.add(Embedding(nb_words, 10, input_length=maxlen_right))
right_branch.add(Convolution1D(10, 3))
right_branch.add(MaxPooling1D(3))
right_branch.add(Flatten())
right_branch.add(Dense(10))
right_branch.add(Activation("sigmoid"))

model = Sequential()
model.add(Merge([left_branch, right_branch], mode="concat"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer='adagrad',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(X_train.shape)
print(Y_train.shape)

model.fit([X_train_left, X_train_right], Y_train,
          nb_epoch=1,
          batch_size=32,
          validation_data=([X_test_left, X_test_right], Y_test))
