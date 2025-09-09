import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model (updated name)
model = load_model(r"my_model.h5", compile=False)

# Load tokenizer
with open(r"tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Prediction function
def make_prediction(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=211)  # adjust if needed
    prediction = model.predict(padded_sequences)
    predicted_class = np.argmax(prediction[0])
    label = "AI" if predicted_class == 1 else "Human"
    ai_prob = prediction[0][1] * 100
    human_prob = prediction[0][0] * 100

    # Highlight text based on label
    if label == "AI":
        highlighted_text = f"<span style='color:#ff4d4d; font-weight:bold; text-shadow:1px 1px 5px #ff9999;'>{text}</span>"
    else:
        highlighted_text = f"<span style='color:#33cc33; font-weight:bold; text-shadow:1px 1px 5px #99ff99;'>{text}</span>"

    return label, ai_prob, human_prob, highlighted_text

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form["user_input"]
        if user_input.strip() == "":
            return render_template("index.html", error="⚠️ Please enter some text.")

        label, ai_prob, human_prob, highlighted_text = make_prediction(user_input)
        return render_template("index.html", 
                               prediction=label,
                               ai_prob=ai_prob,
                               human_prob=human_prob,
                               user_input=highlighted_text)  # return highlighted text
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
