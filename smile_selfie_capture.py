import cv2
import numpy as np
from deepface import DeepFace

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

print(cv2.__version__)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Extract the face region
        try:
            # Analyze emotions
            analysis = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']

            # Display detected emotion
            cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # If smile is detected, capture the selfie
            if emotion == 'happy':
                cv2.imwrite("selfie.jpg", frame)
                print("Selfie Captured!")
                break  # Exit loop after capturing
        except:
            pass

    # Display the webcam feed
    cv2.imshow("Smile to Capture", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
