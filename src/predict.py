import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load model
model = load_model("models/emotion_model.h5")

# Emotion labels (change if your training labels are different)
emotions = [
    "angry",
    "calm",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised"
]

# Load audio file
audio_path = "uploads/sample.wav"

audio, sr = librosa.load(audio_path, res_type='kaiser_fast')

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
mfccs = np.mean(mfccs.T, axis=0)

# Reshape for model
X = np.expand_dims(mfccs, axis=0)

# Predict
prediction = model.predict(X)

predicted_class = np.argmax(prediction)

print("Predicted Emotion:", emotions[predicted_class])