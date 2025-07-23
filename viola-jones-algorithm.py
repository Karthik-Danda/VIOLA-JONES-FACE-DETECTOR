# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 21:57:47 2025

@author: bittu
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import sample
from tqdm import tqdm
import time
start_time = time.time()
face_dir = r"C:\Users\ashwi\OneDrive\Desktop\VIOLA\FACE"
nonface_dir = r"C:\Users\ashwi\OneDrive\Desktop\VIOLA\non-face"
img_size = (24, 24)


def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, img_size)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(gray.astype(np.float64))
                labels.append(label)
    return images, labels


face_images, y1 = load_images_from_folder(face_dir, 1)
nonface_images, y2 = load_images_from_folder(nonface_dir, -1)

X = face_images + nonface_images
y = y1 + y2
num_samples = len(X)

print(f"✅ Loaded {num_samples} images. ({len(y1)} faces, {len(y2)} non-faces)")

# --- Compute integral images ---
integral_images = [cv2.integral(img)[1:, 1:] for img in X]
print("✅ Computed integral images.")

# --- Extract Haar Features 1 to 4 ---
feature_list = []
print("🔄 Extracting Haar Features 1 to 4...")
for II in tqdm(integral_images):
    features = []

    # Feature 1: Two-rectangle horizontal
    for w in range(1, 13):
        for h in range(1, 25):
            for x in range(0, 24 - 2*w + 1):
                for y in range(0, 24 - h + 1):
                    A = II[y:y+h, x:x+w].sum()
                    B = II[y:y+h, x+w:x+2*w].sum()
                    features.append(A - B)

    # Feature 2: Tworectangle vertical
    for w in range(1, 25):
        for h in range(1, 13):
            for x in range(0, 24 - w + 1):
                for y in range(0, 24 - 2*h + 1):
                    A = II[y:y+h, x:x+w].sum()
                    B = II[y+h:y+2*h, x:x+w].sum()
                    features.append(A - B)

    # Feature 3: Threerectangle horizontal
    for w in range(1, 9):
        for h in range(1, 25):
            for x in range(0, 24 - 3*w + 1):
                for y in range(0, 24 - h + 1):
                    A = II[y:y+h, x:x+w].sum()
                    B = II[y:y+h, x+w:x+2*w].sum()
                    C = II[y:y+h, x+2*w:x+3*w].sum()
                    features.append(A - B + C)

    # Feature 4: Fourrectangle checkerboard
    for w in range(1, 13):
        for h in range(1, 13):
            for x in range(0, 24 - 2*w + 1):
                for y in range(0, 24 - 2*h + 1):
                    A = II[y:y+h, x:x+w].sum()
                    B = II[y:y+h, x+w:x+2*w].sum()
                    C = II[y+h:y+2*h, x:x+w].sum()
                    D = II[y+h:y+2*h, x+w:x+2*w].sum()
                    features.append((A + D) - (B + C))

    feature_list.append(features)

print("✅ Finished extracting features.")
end_time = time.time()
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
start_time = time.time()
# --- Convert list of features to matrix ---
X_features = np.array(feature_list)
y_labels = np.array(y1 + y2, dtype=np.int32)


print("✅ Feature matrix shape:", X_features.shape)
print("✅ Labels shape:", y_labels.shape)
# --- Split into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.1, random_state=42)

# --- Train AdaBoost classifier with decision stumps ---
base_estimator = DecisionTreeClassifier(max_depth=1)  # decision stump
adaboost = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=0)
adaboost.fit(X_train, y_train)

# --- Predict on test data ---
y_pred = adaboost.predict(X_test)

# --- Evaluate the model ---
acc = accuracy_score(y_test, y_pred)
end_time = time.time()

print(f"✅ Test Accuracy: {acc * 100:.2f}%\n")

print("🔍 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Face (-1)", "Face (+1)"]))


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Face", "Face"], yticklabels=["Non-Face", "Face"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
print(f"🕒 Training time: {end_time - start_time:.2f} seconds")




test_img_path = r"C:\Users\ashwi\OneDrive\Desktop\VIOLA\file-20250324-56-t810mm.avif"
import time
start_time = time.time()

test_img = cv2.imread(test_img_path)
test_img = cv2.resize(test_img, (24, 24))
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Compute integral image
II = cv2.integral(test_gray)[1:, 1:]
test_features = []

# Feature 1: Two-rectangle horizontal
for w in range(1, 13):
    for h in range(1, 25):
        for x in range(0, 24 - 2*w + 1):
            for y in range(0, 24 - h + 1):
                A = II[y:y+h, x:x+w].sum()
                B = II[y:y+h, x+w:x+2*w].sum()
                test_features.append(A - B)

# Feature 2: Two-rectangle vertical
for w in range(1, 25):
    for h in range(1, 13):
        for x in range(0, 24 - w + 1):
            for y in range(0, 24 - 2*h + 1):
                A = II[y:y+h, x:x+w].sum()
                B = II[y+h:y+2*h, x:x+w].sum()
                test_features.append(A - B)

# Feature 3: Three-rectangle horizontal
for w in range(1, 9):
    for h in range(1, 25):
        for x in range(0, 24 - 3*w + 1):
            for y in range(0, 24 - h + 1):
                A = II[y:y+h, x:x+w].sum()
                B = II[y:y+h, x+w:x+2*w].sum()
                C = II[y:y+h, x+2*w:x+3*w].sum()
                test_features.append(A - B + C)

# Feature 4: Four-rectangle checkerboard
for w in range(1, 13):
    for h in range(1, 13):
        for x in range(0, 24 - 2*w + 1):
            for y in range(0, 24 - 2*h + 1):
                A = II[y:y+h, x:x+w].sum()
                B = II[y:y+h, x+w:x+2*w].sum()
                C = II[y+h:y+2*h, x:x+w].sum()
                D = II[y+h:y+2*h, x+w:x+2*w].sum()
                test_features.append((A + D) - (B + C))
test_features = np.array(test_features).reshape(1, -1)  # 2D for scikit-learn
prediction = adaboost.predict(test_features)

label = "Face (+1)" if prediction[0] == 1 else "Non-Face (-1)"
print(f"🧠 Prediction: {label}")
end_time = time.time()
print(f"🕒 Training time: {end_time - start_time:.2f} seconds")