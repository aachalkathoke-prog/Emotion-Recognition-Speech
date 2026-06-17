import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load dataset
df = pd.read_csv("dataset/processed/features.csv")

# Features and labels
X = df.drop("emotion", axis=1)
y = df["emotion"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert labels to categorical
y_test_cat = to_categorical(y_test)

# Load trained model
model = load_model("models/emotion_model.h5")

# Predict
y_pred = model.predict(X_test)

# Convert predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_classes)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))