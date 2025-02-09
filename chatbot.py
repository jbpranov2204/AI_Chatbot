from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load required data
lemmatizer = WordNetLemmatizer()

with open('training_data/intents.json', 'r') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    if len(intents_list) == 0:
        return "I'm sorry, I don't understand that."

    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return "I'm sorry, I don't understand that."


@app.route('/chat', methods=['POST'])
def chatbot():
    data = request.get_json()
    message = data.get("message")

    if not message:
        return jsonify({"error": "Message parameter is required"}), 400

    intents_list = predict_class(message)
    response = get_response(intents_list, intents)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
