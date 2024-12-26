import streamlit as st
import json
import numpy as np
import random
import nltk
from tensorflow.keras.models import load_model
import pickle

# Load pre-trained model and data
model = load_model('chatbot_model.h5')
with open('words.pkl', 'rb') as f:
    words = pickle.load(f)
with open('classes.pkl', 'rb') as f:
    classes = pickle.load(f)
with open('intents.json') as file:
    intents = json.load(file)

lemmatizer = nltk.WordNetLemmatizer()

# Helper functions
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

def predict_class(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if not intents_list:  # If intents_list is empty
        return "Invalid response, try again."
    tag = intents_list[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

# Streamlit UI
st.title("Chatbot with NLP")
st.markdown("""
This chatbot uses machine learning and NLP techniques to respond to your queries.
Type your message below and get an instant response.
""")

# User input
user_input = st.text_input("You:", "")

# If user submits a message
if st.button("Send"):
    if user_input:
        intents_list = predict_class(user_input)
        response = get_response(intents_list, intents)
        st.text_area("Chatbot:", response, height=200)
    else:
        st.warning("Please enter a message to chat!")
