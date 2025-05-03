import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("../models/sign_language_model.h5")

# Parameters
IMG_SIZE = 64
label_map = {i: chr(i + 65) for i in range(26)}  # 0=A, 1=B, ..., 25=Z
CONFIDENCE_THRESHOLD = 0.9  # Stricter threshold
ROI_SIZE = 300  # Larger ROI to capture signs better

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Background subtraction (optional, uncomment if needed)
# fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip frame to correct mirror effect
    frame = cv2.flip(frame, 1)

    # Define ROI in the center
    height, width = frame.shape[:2]
    start_x = (width - ROI_SIZE) // 2
    start_y = (height - ROI_SIZE) // 2
    roi = frame[start_y:start_y + ROI_SIZE, start_x:start_x + ROI_SIZE]

    # Preprocess ROI
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)
    
    # Predict
    pred = model.predict(img, verbose=0)
    confidence = np.max(pred)
    label_idx = np.argmax(pred)
    label = label_map[label_idx] if confidence >= CONFIDENCE_THRESHOLD else "None"

    # Draw ROI and prediction
    cv2.rectangle(frame, (start_x, start_y), (start_x + ROI_SIZE, start_y + ROI_SIZE), (0, 255, 0), 2)
    text = f"Sign: {label} ({confidence:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 50
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Show frame
    cv2.imshow("Sign Language Recognition", frame)
    
    # Debug: Save ROI for inspection (press 's')
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("roi_debug.jpg", roi)
        print("ROI saved as roi_debug.jpg")

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()