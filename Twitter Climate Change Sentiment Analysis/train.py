import numpy as np
import pandas as pd
import tensorflow as tf
import keras.preprocessing.text as kpt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import json

data = pd.read_csv('processed_data.csv', delimiter=',', header=None, usecols=[0, 1])

train_x = [tweet for tweet in data[1]]
train_y = [label for label in data[0]]

max_words = 3000

# create a new Tokenizer object
tokenizer = kpt.Tokenizer(num_words=max_words)
# feed the tweets to the Tokenizer
tokenizer.fit_on_texts(train_x)

# create the list of words and their IDs
tokens_dict = tokenizer.word_index

# save for later use
with open('tokens_dict.json', 'w') as dictionary_file:
    json.dump(tokens_dict, dictionary_file)


def convert_text_to_index_array(text):
    # one really important thing that `text_to_word_sequence` does
    # is make all texts the same length -- in this case, the length
    # of the longest text in the set.
    return [tokens_dict[word] for word in kpt.text_to_word_sequence(text)]


allWordIndices = []
# for each tweet, change each token to its ID in the Tokenizer's word_index
for text in train_x:
    wordIndices = convert_text_to_index_array(text)
    allWordIndices.append(wordIndices)

# create matrices out of the indexed tweets
train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
# treat the labels as categories
train_y = tf.keras.utils.to_categorical(np.asarray(train_y), 4)

model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_x, train_y,
          batch_size=32,
          epochs=5,
          verbose=1,
          validation_split=0.1,
          shuffle=True)

# saving the net
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

model.save_weights('model.h5')

