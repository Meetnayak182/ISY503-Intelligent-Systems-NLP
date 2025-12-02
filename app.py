# app.py - Simple Flask web app for sentiment analysis

from flask import Flask, request, render_template_string
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

# 1. Configuration 
MAX_LEN = 200   # same as in Colab
MODEL_PATH = "sentiment_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"

# 2. Load model and tokenizer
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer file not found: {TOKENIZER_PATH}")

loaded_model = keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    loaded_tokenizer = pickle.load(f)

# 3. Cleaning function 
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)          
    text = re.sub(r'[^a-z]+', ' ', text)        
    text = re.sub(r'\s+', ' ', text).strip()    
    return text

# 4. Prediction function
def predict_sentiment(text):
    clean = clean_text(text)
    seq = loaded_tokenizer.texts_to_sequences([clean])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = loaded_model.predict(pad, verbose=0)[0][0]
    label = 1 if prob >= 0.5 else 0
    label_str = "Positive review" if label == 1 else "Negative review"
    return label_str, float(prob)

# 5. Flask app setup with a very simple HTML template
app = Flask(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ISY503 Sentiment Analysis</title>
    <style>
        body { font-family: sans-serif; background: #f4f7f6; padding: 20px; color: #333; }
        .container { max-width: 600px; margin: 20px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 5px; }
        p.subtitle { text-align: center; color: #7f8c8d; margin-bottom: 20px; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; resize: vertical; }
        button { width: 100%; padding: 10px; margin-top: 15px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #2980b9; }
        .result-box { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px; border-left: 4px solid #3498db; }
        .result-text { font-size: 1.1em; font-weight: bold; color: #2c3e50; }
        .prob-text { font-size: 0.9em; color: #7f8c8d; }
    </style>
</head>
<body>
<div class="container">
    <h1>Sentiment Analysis</h1>
    <p class="subtitle">ISY503 Assessment 3</p>
    <form method="POST">
        <label for="review_text"><b>Enter Product Review:</b></label>
        <textarea id="review_text" name="review_text" placeholder="Type review here...">{{ input_text }}</textarea>
        <button type="submit">Analyse Sentiment</button>
    </form>
    {% if prediction %}
        <div class="result-box">
            <div class="result-text">Result: {{ prediction }}</div>
            <div class="prob-text">Confidence: {{ probability }}</div>
        </div>
    {% endif %}
</div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None
    input_text = ""

    if request.method == "POST":
        input_text = request.form.get("review_text", "")
        if input_text.strip():
            prediction, prob = predict_sentiment(input_text)
            probability = f"{prob:.3f}"

    return render_template_string(
        HTML_TEMPLATE,
        prediction=prediction,
        probability=probability,
        input_text=input_text
    )

# 6. Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
