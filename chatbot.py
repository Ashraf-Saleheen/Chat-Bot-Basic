import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lematizer = WordNetLemmatizer()
# intents = json.load(open('intents.json').read())
with open('intents.json', 'r') as file:
    intents = json.load(file)


# words = pickle.load(open('words.pickle'),'rb')
with open('words.pickle', 'rb') as file:
    words = pickle.load(file)
# classes = pickle.load(open('classes.pickle'),'rb')
with open('classes.pickle', 'rb') as file:
    classes = pickle.load(file)
model = load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lematizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
                return np.array(bag)
# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHHOLD = 0.25
#     result = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHHOLD]
#
#     result.sort(key=lambda x:x[1], reverse=True)
#     return_list = []
#
#     for r in result:
#         return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
#     return return_list

def predict_class(message):
    bow = generate_bow_from_message(message)
    if bow is None:
        print("Error: Bow is None")
        return None

    res = model.predict(np.array([bow]))[0]
    return res
# data_processing.py

def generate_bow_from_message(message):
    # Your code to generate the bag-of-words representation
    # Return the generated bow
    pass


def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result = random.choice(i['responses'])
            break
    return result
print('Go Bot is Running!')

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
