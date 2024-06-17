import pickle
import json
import random
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# Load intents data from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
model = load_model('chatbot_model.h5') 

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    #print(sentence_words)
    bag =[0]*len(words)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    #print(f"Bag of words: {bag}")  # Debug print
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    #print(f"Bag of words: {bow}")
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent': classes[r[0]],'probability':str(r[1])})
    #print(return_list)
    return return_list 


def get_response(intents_list,intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    #print(result)
    return result
    
print("The BOT is ready to chat with you!")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints,intents)
    print(res)


                                  



