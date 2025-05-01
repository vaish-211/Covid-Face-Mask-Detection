import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import os
from datetime import datetime

# Parse command-line arguments (optional)
parser = argparse.ArgumentParser(description='Real-time face mask detection')
parser.add_argument('--model', type=str, choices=['deep', 'transfer', 'tuned'],
                    help='Model to use: deep, transfer, or tuned')
parser.add_argument('--video', type=str,
                    help='Path to input video file')
args = parser.parse_args()

# Model selection
model_paths = {
    'deep': '../models/model_deep.h5',
    'transfer': '../models/model_transfer.h5',
    'tuned': '../models/model_tuned.h5'
}
if args.model:
    model_choice = args.model
else:
    print("Select model (enter number): 1=deep, 2=transfer, 3=tuned")
    while True:
        choice = input("Choice [2]: ").strip() or '2'
        try:
            model_choice = ['deep', 'transfer', 'tuned'][int(choice)-1]
            break
        except (ValueError, IndexError):
            print("Invalid input. Enter 1, 2, or 3.")
model_path = model_paths[model_choice]
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found.")
model = load_model(model_path)
print(f"Using model: {model_choice}")

# Video selection
if args.video:
    video_path = args.video
else:
    print("Enter video file name (e.g., video.mp4) or press Enter for webcam")
    video_path = input("Video: ").strip()
    video_path = os.path.join('../Sample Pictures', video_path) if video_path else 0
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError(f"Error opening video source: {video_path}")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
if face_cascade.empty():
    raise FileNotFoundError("Haar Cascade file not found.")

# Initialize output video writer
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_path = os.path.join('..', 'Sample Pictures', 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150))
        face = img_to_array(face)
        face = face / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Predict mask
        pred = model.predict(face)[0][0]
        label = 'Mask' if pred < 0.5 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        score_text = f'{pred:.2f}'
        
        # Annotate frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({score_text})", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Write frame to output video
    out.write(frame)
    
    # Display frame
    cv2.imshow('Face Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Output video saved to {out_path}")