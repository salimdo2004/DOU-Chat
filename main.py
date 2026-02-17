import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import random
import string
from nltk.tokenize import word_tokenize
from symspellpy import SymSpell, Verbosity
import json
import time
from tensorflow import keras


# Charger le modèle Keras
model2 = tf.keras.models.load_model("chatbot_model.keras")



# Initialisation de SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "fr-100k.txt"
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Initialiser Flask
app = Flask(__name__)

# Variable pour stocker les questions déjà posées et leurs réponses avec timestamp
user_memory = {}


# Charger les intents
def load_intents():
    with open('data.json', 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents


# Créer l'index inversé
def create_index(intents):
    index = {}
    for intent in intents['intents']:
        if 'intent' in intent:
            for question in intent['questions']:
                for word in question.split():
                    word = word.lower()
                    if word not in index:
                        index[word] = []
                    index[word].append((intent['intent'], question))
    return index


def find_response(user_input, index, intents):
    words = user_input.lower().split()
    possible_intents = {}

    for word in words:
        if word in index:
            for tag, question in index[word]:
                possible_intents.setdefault(tag, []).append(question)

    if possible_intents:
        best_match = max(possible_intents, key=lambda tag: len(possible_intents[tag]))

        for intent in intents['intents']:
            if intent.get('intent') == best_match:
                responses = intent.get('responses')
                if isinstance(responses, list) and len(responses) > 0:
                    return random.choice(responses)
                else:
                    print("⚠ Intent trouvé sans réponses :", intent)
                    return "Problème de configuration des réponses."

    return "Désolé, je ne comprends pas."


# Variables du modèle
tokens = ["token1", "token2"]  # Remplacez avec vos vrais tokens
lemmatizer = None


def input_presentation(text):
    t = word_tokenize(text)
    vect_x = [0] * len(tokens)
    for item in t:
        if item not in string.punctuation:
            result = lemmatizer.lemmatize(item.lower()) if lemmatizer else item.lower()
            if result in tokens:
                idx = tokens.index(result)
                vect_x[idx] = 1
    return vect_x


def pred_f(text, index, intents):
    response = find_response(text, index, intents)
    if response != "Désolé, je ne comprends pas.":
        return response, 100

    if input_presentation(text) == [0] * len(tokens):
        return "Je ne comprends pas", 0

    prediction = model2.predict(np.array(input_presentation(text)).reshape(1, -1))
    idx = np.argmax(prediction)
    responses = [["réponse 1", "réponse 2"]]  # À personnaliser
    return random.choice(responses[idx]), max(prediction[0]) * 100


def correct_text(text):
    words = text.split()
    corrected_words = []
    for word in words:
        suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        corrected_words.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected_words)


# Page d’accueil
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json.get("text", "").strip().lower()
        current_time = time.time()

        if user_input == "":
            return jsonify({"response": "Veuillez écrire un message.", "score": 0})

        corrected_input = correct_text(user_input)

        # Mémoire utilisateur
        if user_input in user_memory:
            old_response, last_time = user_memory[user_input]
            if current_time - last_time < 60:
                return jsonify({"response": old_response, "score": "réponse récente"})

        # Prédiction
        response, score = pred_f(corrected_input, index, intents)

        # Si faible confiance → enregistrer la question
        if score < 50:
            try:
                with open("historique_questions.txt", "a", encoding="utf-8") as f:
                    f.write(user_input + "\n")
            except Exception as e:
                print("Erreur fichier :", e)

            response = "Je n’ai pas bien compris votre demande. Pouvez-vous reformuler ?"

        # Sauvegarde mémoire
        user_memory[user_input] = (response, current_time)

        return jsonify({"response": response, "score": score})

    except Exception as e:
        print("Erreur serveur :", e)

        # Enregistrer aussi la question en cas d'erreur technique
        try:
            with open("historique_questions.txt", "a", encoding="utf-8") as f:
                f.write("ERREUR : " + request.json.get("text", "") + "\n")
        except:
            pass

        return jsonify({
            "response": "Un problème technique est survenu. Merci de réessayer dans quelques instants.",
            "score": 0
        })
# Charger les données à l'avance
intents = load_intents()
index = create_index(intents)

# Lancer le serveur
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=8006)
