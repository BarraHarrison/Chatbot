# This is the main file for the chatbot
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf  

# Load resources
lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = tf.keras.models.load_model('chatbot_model.h5')  

# Functions
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence.lower())  
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        for intent in intents_json['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
    return "I'm not sure I understand. Can you rephrase?"

# Main loop
print("Chatbot is running! Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print("Goodbye!")
        break

    intents_list = predict_class(message)
    response = get_response(intents_list, intents)
    print(f"Chatbot: {response}")
