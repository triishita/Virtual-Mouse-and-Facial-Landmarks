import cv2
import dlib

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from dlib's website

# Function to detect landmarks on faces
def detect_landmarks(gray_img, face):
    shape = predictor(gray_img, face)
    landmarks = []
    for i in range(68):
        landmarks.append((shape.part(i).x, shape.part(i).y))
    return landmarks

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for better processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Get facial landmarks for each face
        landmarks = detect_landmarks(gray, face)

        # Draw landmarks on the frame
        for landmark in landmarks:
            cv2.circle(frame, landmark, 1, (0, 255, 0), -1)

    # Display the frame with detected landmarks
    cv2.imshow('Face Landmark Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()