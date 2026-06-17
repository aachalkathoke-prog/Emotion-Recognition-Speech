from flask import Flask, render_template, request
import os
import librosa
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model = load_model("models/emotion_model.h5")

# Load emotion labels
df = pd.read_csv("dataset/processed/features.csv")

encoder = LabelEncoder()
encoder.fit(df["emotion"])


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)

    mfccs = np.mean(
        librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T,
        axis=0
    )

    return mfccs.reshape(1, -1)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":

        if "file" not in request.files:
            return render_template(
                "index.html",
                prediction="No file selected"
            )

        file = request.files["file"]

        if file.filename == "":
            return render_template(
                "index.html",
                prediction="No file selected"
            )

        os.makedirs("uploads", exist_ok=True)

        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        try:
            features = extract_features(filepath)

            pred = model.predict(features)

            predicted_class = np.argmax(pred)

            prediction = encoder.inverse_transform(
                [predicted_class]
            )[0]

            filename = file.filename

            print("Predicted Emotion:", prediction)

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        filename=filename
    )


if __name__ == "__main__":
    app.run(debug=True)