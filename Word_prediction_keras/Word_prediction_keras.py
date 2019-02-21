'''This is a word prediction model built with LSTM. It takes 5 input sequence and 
predict the next word in from the sequence. 

For creating dictionary/tokenizing word through keras library:
https://keras.io/preprocessing/text/#tokenizer

Some code has been taken from:
https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_lstm.py

To understand Embedding layer:
https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/

To understand LSTM:
https://keras.io/getting-started/sequential-model-guide/
https://adventuresinmachinelearning.com/keras-lstm-tutorial/

'''

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import array
import collections
import os

print(tf.__version__)

data_path = "/Users/sadieee04/Documents/gitproject/Machine_Learning/Word_prediction_keras/"

def read_words(filename):
    #print("filename", filename)
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_data():
    # get the data paths
    train_path = os.path.join(data_path, "short.ptb.train.txt")
    valid_path = os.path.join(data_path, "short.ptb.valid.txt")
    test_path = os.path.join(data_path, "short.ptb.test.txt")

    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))

    return train_data, valid_data, test_data, vocabulary, reversed_dictionary

#Loading data from text file and creating the dictionary
train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data()

#Creates data label for supervised training
def create_target_data():
    #Every word predicts the next word hence the target data is 1 index shifted from training data
    train_label = train_data[1:]
    train_label.append(0)

    valid_label = valid_data[1:]
    valid_label.append(0)

    test_label = test_data[1:]
    test_label.append(0)

    print("train data length", len(train_data))
    print("train label length",len(train_label))

    print("train data [:5]",train_data[:5])
    print("train label [:5]",train_label[:5])

    return train_label, valid_label, test_label

#Creates data label for supervised training
train_label, valid_label, test_label = create_target_data()

print(train_label[:5])
print("total vocabulary", vocabulary)

#t1---t5
timesteps = 5 

#size of the vector dimension coming out from embedding layer to represent each word
# "example" = [2 4 6] for hidden_size = 3
hidden_size = 150  

#Reshaping data to make it compatible with the input layers
def reshape_data(train_data, train_label, test_data, test_label, valid_data, valid_label):
    train_data = np.array(train_data)
    train_data = np.pad(train_data, (0, timesteps-len(train_data)%timesteps), 'constant')
    train_data = np.reshape(train_data, (int(len(train_data)/timesteps), timesteps))

    train_label = np.array(train_label)
    train_label = np.pad(train_label, (0, timesteps-len(train_label)%timesteps), 'constant')
    train_label = np.reshape(train_label, (int(len(train_label)/timesteps), timesteps))

    test_data = np.array(test_data)
    test_data = np.pad(test_data, (0, timesteps-len(test_data)%timesteps), 'constant')
    test_data = np.reshape(test_data, (int(len(test_data)/timesteps), timesteps))

    test_label = np.array(test_label)
    test_label = np.pad(test_label, (0, timesteps-len(test_label)%timesteps), 'constant')
    test_label = np.reshape(test_label, (int(len(test_label)/timesteps), timesteps))

    valid_data = np.array(valid_data)
    valid_data = np.pad(valid_data, (0, timesteps-len(valid_data)%timesteps), 'constant')
    valid_data = np.reshape(valid_data, (int(len(valid_data)/timesteps), timesteps))

    valid_label = np.array(valid_label)
    valid_label = np.pad(valid_label, (0, timesteps-len(valid_label)%timesteps), 'constant')
    valid_label = np.reshape(valid_label, (int(len(valid_label)/timesteps), timesteps))

    print("train data shape", train_data.shape)
    print("train label shape", train_label.shape)

    return train_data, train_label, test_data, test_label, valid_data, valid_label

#Reshaping data to make it compatible with the input layers
train_data, train_label, test_data, test_label, valid_data, valid_label = reshape_data(train_data, train_label, test_data, test_label, valid_data, valid_label)

model = keras.Sequential()
model.add(keras.layers.Embedding(vocabulary, hidden_size, input_length=timesteps))
model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
model.add(keras.layers.LSTM(hidden_size, return_sequences=True))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(timesteps, activation='relu'))

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(train_data, train_label, epochs=600)

input_text = np.array([140, 162, 165, 176, 181])
input_text = input_text.reshape(1, 5)
print(input_text.shape)
print(model.predict(test_data))

#Creat graph of accuracy and loss over time
'''history_dict = history.history
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

plt.show()'''