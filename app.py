from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import socket
import re

# -------------------- Load Model and Vectorizer --------------------
try:
    lgbm_model = joblib.load("sentiment_model.pkl")
    tfidf_vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model or vectorizer: {e}")

# Sentiment label mapping
class_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# -------------------- Preprocessing Function --------------------
def preprocess_comment(comment):
    comment = comment.lower()
    comment = re.sub(r"http\S+|www\S+|https\S+", '', comment, flags=re.MULTILINE)  # Remove URLs
    comment = re.sub(r'\W', ' ', comment)  # Remove special characters
    comment = re.sub(r'\s+', ' ', comment).strip()  # Remove extra spaces
    return comment

# -------------------- Flask App --------------------
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Handle GET requests gracefully
    if request.method == 'GET':
        return jsonify({
            "message": "This endpoint only supports POST for predictions. "
                       "Send a POST request with JSON {\"comment\": \"your text\"}."
        }), 200

    # Handle POST requests for prediction
    try:
        # Get comment from JSON or form data
        if request.is_json:
            data = request.get_json()
            comment = data.get('comment', '')
        else:
            comment = request.form.get('comment', '')

        if not comment:
            return jsonify({'error': 'No comment provided'}), 400

        # Preprocess and vectorize
        cleaned_comment = preprocess_comment(comment)
        comment_tfidf = tfidf_vectorizer.transform([cleaned_comment])

        # Predict probabilities and sentiment
        proba = lgbm_model.predict_proba(comment_tfidf)
        pred = int(np.argmax(proba))  # Always 0,1,2
        sentiment = class_labels[pred]
        confidence = float(np.max(proba))

        return jsonify({
            'comment': comment,
            'sentiment': sentiment,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Find a free port starting at 5000
    port = 5000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while s.connect_ex(('127.0.0.1', port)) == 0:
        port += 1
    s.close()

    print(f"Starting Flask server on port {port}")
    app.run(debug=True, host='0.0.0.0', port=port)
