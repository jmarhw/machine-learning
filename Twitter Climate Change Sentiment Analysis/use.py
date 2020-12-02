import json
import numpy as np
import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import pandas as pd

tokenizer = Tokenizer(num_words= 3000)
labels = ['does not believe in man-made climate change', #0
          'neither supports nor refutes the belief of man-made climate change', #1
          'supports the belief of man-made climate change', #2
          'links to factual news about climate change'] #3

# read in saved dictionary
with open('tokens_dict.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

# this utility makes sure that all the words in the test input
# are registered in the dictionary
# before trying to turn them into a matrix.
def convert_text_to_index_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word]) #it ignores words not in scope
    return wordIndices

# read in saved model structure
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# and create a model from that
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

def prediction(tweet, model):
    converted_tweet = convert_text_to_index_array(tweet)
    tokenized_tweet = tokenizer.sequences_to_matrix([converted_tweet], mode = 'binary')
    return model.predict(tokenized_tweet)

test_tweets = pd.read_csv("test_data.csv", header = None, usecols=[0,1])
processed_tweets = [tweet for tweet in test_tweets[0]]
tweets = [tweet for tweet in test_tweets[1]]

for i in range(0,len(tweets)):
    pred = prediction(processed_tweets[i], model)
    print(f"The tweet: \'{tweets[i]}\'")
    print("%s with %f%% confidence \n" % (labels[np.argmax(pred)], pred[0][np.argmax(pred)] * 100))