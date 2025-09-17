import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model


try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)

except Exception:
    pass

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents JSON
with open("intents.json", "r") as file:
    intents = json.load(file)

# Load trained data
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbot_inimodel.h5")


# ------------------ Helper functions ------------------ #
def clean_up_sentence(sentence):
    """Tokenize and lemmatize a sentence"""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    """Convert a sentence into a bag of words"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    """Predict the intent of a given sentence"""
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]

    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """Pick a random response based on predicted intent"""
    if not intents_list:  # If no intent passes threshold
        return "I'm not sure I understand. Can you rephrase?"

    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Sorry, I don’t know how to respond to that."


# ------------------ Chatbot loop ------------------ #
print("Great! Bot is running! (type 'quit' to exit)\n")

while True:
    message = input("You: ")
    if message.lower() == "quit":
        print("Bot: Goodbye!")
        break

    ints = predict_class(message)
    res = get_response(ints, intents)
    print("Bot:", res)
