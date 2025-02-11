import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Define virtual screen size
VIRTUAL_SCREEN_WIDTH = 1920
VIRTUAL_SCREEN_HEIGHT = 1080    

# Define thresholds for movement sensitivity
WAVE_THRESHOLD = 150  # Distance to consider as a wave (75 for tapping, 225 for a proper wave)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables to track previous position of the index finger
previous_index_x = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and find hands
    results = hands.process(rgb_frame)

    # Get the hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Check if the hand is left
            if hand_info.classification[0].label == 'Left':
                # Get the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = int(index_finger_tip.x * VIRTUAL_SCREEN_WIDTH)

                # Check for waving to the left
                if previous_index_x is not None and abs(previous_index_x - index_x) > WAVE_THRESHOLD:
                    if index_x < previous_index_x:  # Waving to the left
                        # Minimize all tabs (this will minimize the active window)
                        pyautogui.hotkey('alt', 'tab')  # Show desktop (minimize all)

                # Update previous index finger position
                previous_index_x = index_x

                # Draw hand landmarks on the frame (optional, can be removed if not needed)
                # mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame (commented out to remove the camera tab)
    # cv2.imshow('Finger Tracking', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows  
cap.release()
cv2.destroyAllWindows()