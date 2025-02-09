import random
import json
import sys
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
with open('training_data/intents.json', 'r') as file:
    intents = json.load(file)

def doc():
    words = []
    classes = []
    documents = []
    ignoreLetters = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each pattern into words
            wordList = nltk.word_tokenize(pattern)
            words.extend(wordList)
            # Add documents as tuples of (tokenized_words, intent_tag)
            documents.append((wordList, intent['tag']))
            # Add intent tag to classes if not already present
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    lemmatizer = WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignoreLetters]
    words = sorted(set(words))

    classes = sorted(set(classes))

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    outputEmpty = [0] * len(classes)

    for document in documents:
        bag = []
        wordPatterns = document[0]
        wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
        for word in words:
            bag.append(1) if word in wordPatterns else bag.append(0)

        outputRow = list(outputEmpty)
        outputRow[classes.index(document[1])] = 1
        training.append(bag + outputRow)

    random.shuffle(training)
    training = np.array(training)  # Convert training list to numpy array

    trainX = training[:, :len(words)]
    trainY = training[:, len(words):]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(128, input_shape=(trainX.shape[1],), activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

    sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5')

    print('Done')

if __name__ == "__main__":
    doc()
