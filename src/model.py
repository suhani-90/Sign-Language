import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Function to load data
def load_data(data_path):
    images = []
    labels = []
    label_map = {chr(i): i - 65 for i in range(65, 91)}  # A=0, B=1, ..., Z=25
    
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):  # Skip if not a directory
            continue
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:  # Skip if image can't be loaded
                continue
            img = img / 255.0  # Normalize to [0, 1]
            images.append(img)
            labels.append(label_map[folder])
    
    return np.array(images), np.array(labels)

# Function to build the CNN-LSTM model
def build_model(input_shape=(64, 64, 3), num_classes=26):
    model = models.Sequential([
        # CNN layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        
        # Reshape for LSTM
        layers.Reshape((-1, 128)),  # Adjust based on CNN output
        layers.LSTM(64, return_sequences=False),
        
        # Dense layers
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Main execution
if __name__ == "__main__":
    # Define paths
    train_path = "../dataset/processed/asl_alphabet_train"
    test_path = "../dataset/processed/asl_alphabet_test"

    # Load data
    print("Loading training data...")
    X_train, y_train = load_data(train_path)
    print("Loading test data...")
    X_test, y_test = load_data(test_path)

    # Split training data into train and validation sets
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Print shapes for verification
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

    # Build the model
    print("Building model...")
    model = build_model()
    model.summary()

    # Train the model
    print("Training model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc}")

    # Save the model
    print("Saving model...")
    model.save("../models/sign_language_model.h5")
    print("Model saved successfully!")

   