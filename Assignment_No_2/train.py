import os
import cv2
import face_recognition
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


dataset_path = "dataset"
positive_path = os.path.join(dataset_path, "positive")
negative_path = os.path.join(dataset_path, "negative")

X = []
y = []

# Function to get embeddings
def get_embeddings_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is None:
            continue
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if len(encodings) > 0:
            X.append(encodings[0])
            y.append(label)

# Positive samples = 1, Negative samples = 0
get_embeddings_from_folder(positive_path, 1)
get_embeddings_from_folder(negative_path, 0)

X = np.array(X)
y = np.array(y)


model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl")
