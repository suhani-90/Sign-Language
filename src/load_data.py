import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(data_path):
    images = []
    labels = []
    label_map = {chr(i): i-65 for i in range(65, 91)}  # A=0, B=1, ..., Z=25
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path) / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(label_map[folder])
    
    return np.array(images), np.array(labels)

# Load processed train and test data
train_path = "../dataset/processed/asl_alphabet_train"
test_path = "../dataset/processed/asl_alphabet_test"

X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Split train data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")