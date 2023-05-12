import random
import json
import pickle
import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))

# New code to update the classes list
classes = []
for intent in intents['intents']:
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

pickle.dump(classes, open('classes.pkl', 'wb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)

def build_model(input_shape, output_shape, lr=0.01, hidden_units=16, dropout=0.2):
    model = Sequential()
    model.add(Dense(hidden_units, input_shape=(input_shape,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_units))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_training_data(intents, words, classes):
    X = []
    Y = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            bag = bag_of_words(pattern, words)
            X.append(bag)
            output = [0] * len(classes)
            output[classes.index(intent['tag'])] = 1
            Y.append(output)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y

X, Y = create_training_data(intents, words, classes)
input_shape = X.shape[1]
output_shape = len(classes)
model = build_model(input_shape, output_shape)

model.fit(X, Y, epochs=100, batch_size=8, verbose=1)
model.save('chatbotmodel.h5')
