import os
import librosa
import numpy as np
import pandas as pd

dataset_path = "dataset"

features = []
labels = []

emotion_map = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful"
}

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]

            if emotion_code in emotion_map:

                file_path = os.path.join(root, file)

                audio, sr = librosa.load(file_path, sr=None)

                mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

                mfcc_scaled = np.mean(mfcc.T, axis=0)

                features.append(mfcc_scaled)
                labels.append(emotion_map[emotion_code])

df = pd.DataFrame(features)
df["emotion"] = labels

os.makedirs("dataset/processed", exist_ok=True)

df.to_csv("dataset/processed/features.csv", index=False)

print("Feature extraction completed!")