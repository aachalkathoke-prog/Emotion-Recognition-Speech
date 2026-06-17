import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load features
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

print("Preprocessing completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

