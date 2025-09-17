import random
import json
import pickle
import numpy as np
import tensorflow as tf
import re

import nltk
from nltk.stem import WordNetLemmatizer

# try to download punkt, but if it fails use a fallback tokenizer
try:
    nltk.download("punkt", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    pass

lemmatizer = WordNetLemmatizer()

# intents = json.load(open("intents.json").read())
with open("intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)


words = []
classes = []
documents = []
ignoreLetters = ["?", "!", ".", ","]


def simple_tokenize(text):
    # a small fallback tokenizer
    return re.findall(r"\b\w+'?\w*\b", text.lower())


for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        try:
            wordList = nltk.word_tokenize(pattern)
        except LookupError:
            wordList = simple_tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

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
training = np.array(training)

X_train = training[:, : len(words)]
Y_train = training[:, len(words) :]

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, input_shape=(len(X_train[0]),), activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(len(Y_train[0]), activation="softmax"))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
hist = model.fit(
    np.array(X_train), np.array(Y_train), epochs=200, batch_size=5, verbose=1
)
model.save("chatbot_inimodel.h5")
print("Training complete and model saved to chatbot_ini_model.h5")
