import cv2
import dlib
import mediapipe as mp
import math
import pyautogui
from enum import IntEnum
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbcontrol

pyautogui.FAILSAFE = False
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Gesture Encodings 
class Gest(IntEnum):
    FIST = 0
    PINKY = 1
    RING = 2
    MID = 4
    LAST3 = 7
    INDEX = 8
    FIRST2 = 12
    LAST4 = 15
    THUMB = 16    
    PALM = 31
    V_GEST = 33
    TWO_FINGER_CLOSED = 34
    PINCH_MAJOR = 35
    PINCH_MINOR = 36

# Multi-handedness Labels
class HLabel(IntEnum):
    MINOR = 0
    MAJOR = 1

# Convert Mediapipe Landmarks to recognizable Gestures
class HandRecog:
    
    def __init__(self, hand_label):
        self.finger = 0
        self.ori_gesture = Gest.PALM
        self.prev_gesture = Gest.PALM
        self.frame_count = 0
        self.hand_result = None
        self.hand_label = hand_label

    def update_hand_result(self, hand_result):
        self.hand_result = hand_result

    def get_gesture(self):
        # Simplified gesture recognition logic
        if not self.hand_result:
            return Gest.PALM
        # Assuming Index finger up is a gesture we want to detect
        tip_ids = [4, 8, 12, 16, 20]
        if self.hand_result.landmark[tip_ids[1]].y < self.hand_result.landmark[tip_ids[1] - 2].y:
            return Gest.INDEX
        return Gest.PALM

class Controller:
    @staticmethod
    def handle_controls(gesture, hand_result, image):
        h, w, _ = image.shape
        if gesture == Gest.INDEX:
            # Move cursor
            for id, lm in enumerate(hand_result.landmark):
                if id == 8:  # Index finger tip
                    x, y = int(lm.x * w), int(lm.y * h)
                    pyautogui.moveTo(x, y)
        elif gesture == Gest.FIST:
            # Click
            pyautogui.click()

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_landmarks(image, gray_img):
    faces = detector(gray_img)
    for face in faces:
        shape = predictor(gray_img, face)
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

def main():
    cap = cv2.VideoCapture(0)
    hand_recog_major = HandRecog(HLabel.MAJOR)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            results = hands.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            detect_landmarks(image, gray)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_recog_major.update_hand_result(hand_landmarks)
                    gesture = hand_recog_major.get_gesture()
                    Controller.handle_controls(gesture, hand_landmarks, image)

            cv2.imshow('Gesture and Face Landmark Controller', image)
            if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
