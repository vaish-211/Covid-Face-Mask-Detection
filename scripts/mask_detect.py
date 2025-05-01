import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model = load_model(os.path.join('..', 'models', 'model.h5'))

# Image dimensions
img_width, img_height = 150, 150

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(os.path.join('..', 'haarcascade_frontalface_default.xml'))
if face_cascade.empty():
    raise ValueError("Error loading Haar Cascade file. Check the path to 'haarcascade_frontalface_default.xml'.")

# Initialize video capture
cap = cv2.VideoCapture(os.path.join('..', 'Sample Pictures', 'video.mp4'))
if not cap.isOpened():
    raise ValueError("Error opening video file. Check if 'video.mp4' exists in 'Sample Pictures'.")

# Create input directory if it doesn't exist
os.makedirs(os.path.join('..', 'input'), exist_ok=True)

# Initialize variables
img_count_full = 0
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

while True:
    img_count_full += 1
    response, color_img = cap.read()
    
    if not response:
        break
    
    # Resize image
    scale = 50
    width = int(color_img.shape[1] * scale / 100)
    height = int(color_img.shape[0] * scale / 100)
    dim = (width, height)
    color_img = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)

    # Convert to grayscale for face detection
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=6)

    img_count = 0
    for (x, y, w, h) in faces:
        img_count += 1
        # Extract face region
        color_face = color_img[y:y+h, x:x+w]
        
        # Save face image
        face_path = os.path.join('..', 'input', f'{img_count_full}_{img_count}_face.jpg')
        cv2.imwrite(face_path, color_face)
        
        # Preprocess image for prediction
        img = load_img(face_path, target_size=(img_width, img_height))
        img = img_to_array(img)
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)
        
        # Predict
        prediction = model.predict(img)
        if prediction[0][0] < 0.5:
            class_label = "Mask"
            color = (255, 0, 0)  # Blue for Mask
        else:
            class_label = "No Mask"
            color = (0, 255, 0)  # Green for No Mask

        # Draw rectangle and text
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        org = (x, y-10)  # Place text above face
        cv2.putText(color_img, class_label, org, font, fontScale, color, thickness, cv2.LINE_AA)

    # Display video
    cv2.imshow('Face Mask Detection', color_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
