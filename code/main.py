import random
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import pickle
import neattext as nt
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load the trained model
pipe_lr = joblib.load(
open("moodify_emotion_classifier_naive_bayes.pkl", "rb"))
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
emotions_emoji_dict = {
"anger": " ",
"disgust": " ",
"fear": " ",
"happy": " ",
"joy": " ",
"neutral": " ",
"sad": " ",
"sadness": " ",
"shame": " ",
"surprise": " "
}
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data['text']
    prediction = predict_emotions(raw_text)
    probability = get_prediction_proba(raw_text)
    df_moods = pd.read_csv("data/data_moods_new.csv")
    status = 'Success'
    classes = pipe_lr.classes_.tolist()
# Filter songs based on the matching mood
    matching_songs_df = df_moods[df_moods['mood'] == prediction]
    matching_songs = []
    num_songs_to_send = 5 # Specify the number of songs to send
    if len(matching_songs_df) > 0:
        matching_songs_sample = matching_songs_df.sample(n=num_songs_to_send)
        for index, row in matching_songs_sample.iterrows():
            song_info = {
            "name": row['name'],
            "artist": row['artist']
            }
            matching_songs.append(song_info)
    response = {
    'text': raw_text,
    'emotion': prediction,
    'emoji': emotions_emoji_dict[prediction],
    'probability': float(np.max(probability)),
    'matching_songs': matching_songs,
    'status': status,
    'classes': classes
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)