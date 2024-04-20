import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the sample image of yourself
sample_image = cv2.imread('WhatsApp Image 2024-04-20 at 2.50.14 PM.jpeg')
sample_gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)

# Extract features from the sample image
sample_face = face_cascade.detectMultiScale(sample_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
if len(sample_face) == 1:
    x, y, w, h = sample_face[0]
    sample_face_roi = sample_gray[y:y+h, x:x+w]

# Start capturing video from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Continuously capture and process frames from the camera
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Perform face recognition by comparing detected faces with the sample image
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Compare the detected face with the sample face using histogram comparison
        if len(sample_face) == 1:
            # Compute histograms of the faces
            hist_sample = cv2.calcHist([sample_face_roi], [0], None, [256], [0, 256])
            hist_face = cv2.calcHist([face_roi], [0], None, [256], [0, 256])

            # Calculate the correlation between histograms
            correlation = cv2.compareHist(hist_sample, hist_face, cv2.HISTCMP_CORREL)

            # If correlation is high, it's likely the same person
            if correlation > 0.7:
                # Draw a rectangle around the detected face and display your name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, "VIgnesh", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                # Draw a rectangle around the detected face and display "Unknown"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the processed frame with rectangles and names
    cv2.imshow('Face Recognition', frame)

    # Wait for the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
