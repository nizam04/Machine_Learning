'''Text classification with movie reviews from tensorflow tutorial. This model predicts a 
probability of value between 0 and 1 for each movie review'''

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import array

print(tf.__version__)

#Loading imdb data from keras. Every integer corresponds to one word
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#Exploring the data
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print(train_data[0])
print("Printing the length of 1st two reviews")
print(len(train_data[0]), len(train_data[1]))

'''Converting the integers back to words'''
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

#Reversing the dictionary (key, value) -> (value, key)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#Getting words from number
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print("First review is......")
print(decode_review(train_data[0]))

#Preparing the data. 
#Each review is being made same word length of 256 by padding 0s
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

#len(train_data[0]), len(train_data[1]) = (256, 256)      

# input shape is the vocabulary count used for the movie reviews (10,000 words)

#Building the model
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

#Loss function and optimzer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#Create validation set
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

#Train the model
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
#Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

'''Predicting reviews with the model
>>> word_index['good']
52
>>> word_index['movie']
20
>>> word_index['bad']
78
.
.
.
Review1 = [52, 20] # "Good movie" 
Review2 = [78, 20] # "Bad movie"
Review3 = [24, 38, 52] # "Not so good"
Review3 = [12, 16, 864] # "It was okay"'''

print("Predicting with the model")
my_test_data = array ([[52, 20, 0], [78, 20, 0], [24, 38, 52], [12, 16, 864]])
print(model.predict(my_test_data))

#Creat graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

'''plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()'''