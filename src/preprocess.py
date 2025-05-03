import os
import cv2
import numpy as np

# Define paths
base_path = "../dataset/ASL_Alphabet_Dataset"
train_path = os.path.join(base_path, "asl_alphabet_train")
test_path = os.path.join(base_path, "test_organized")
output_path = "../dataset/processed"
os.makedirs(output_path, exist_ok=True)

# Parameters
IMG_SIZE = 64  # Standard size for resizing
alphabets = [chr(i) for i in range(65, 91)]  # A-Z
ignore_folders = ["space", "nothing", "del"]

def preprocess_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for folder in os.listdir(input_folder):
        if folder in ignore_folders or folder not in alphabets:
            continue  # Skip non-alphabet folders
        folder_path = os.path.join(input_folder, folder)
        output_folder_path = os.path.join(output_folder, folder)
        os.makedirs(output_folder_path, exist_ok=True)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)  # Read in color (BGR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize to 64x64
            output_img_path = os.path.join(output_folder_path, img_name)
            cv2.imwrite(output_img_path, img)  # Save processed image

# Preprocess train and test data
print("Processing training data...")
preprocess_images(train_path, os.path.join(output_path, "asl_alphabet_train"))

print("Processing test data...")
preprocess_images(test_path, os.path.join(output_path, "asl_alphabet_test"))

print("Preprocessing complete!")