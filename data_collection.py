import cv2
import numpy as np
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    print("Downloading Pose Landmarker model...")
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def draw_landmarks(image, landmarks):
    h, w, c = image.shape
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt = (int(end_lm.x * w), int(end_lm.y * h))
            cv2.line(image, start_pt, end_pt, (245, 66, 230), 2)
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 3, (245, 117, 66), cv2.FILLED)

def extract_keypoints(detection_result):
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        # Get x, y, z, visibility (defaulting to 1.0 if not present)
        res = np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)] for lm in landmarks]).flatten()
    else:
        res = np.zeros(33*4)
    return res

# Configuration
DATA_PATH = os.path.join('data')
ACTIONS = np.array(['Out', 'Six', 'Four', 'Wide', 'No_Ball', 'Idle'])
no_sequences = 30 # Number of sequences to collect per action
sequence_length = 30 # Frames per sequence

# Create directories
for action in ACTIONS: 
    os.makedirs(os.path.join(DATA_PATH, action), exist_ok=True)

print("Starting Data Collection Setup...")
cap = cv2.VideoCapture(0)

for action in ACTIONS:
    print(f"Ready to collect data for: {action}. Press 's' to start.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: continue
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            draw_landmarks(frame, detection_result.pose_landmarks[0])
            
        cv2.putText(frame, f'Press "s" to start collecting: {action}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Umpire Signal Data Collection', frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('s'):
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            ret, frame = cap.read()
            if not ret: continue
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = detector.detect(mp_image)
            
            if detection_result.pose_landmarks:
                draw_landmarks(frame, detection_result.pose_landmarks[0])
            
            if frame_num == 0: 
                cv2.putText(frame, 'STARTING SEQUENCE...', (120,200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'{action} | Sequence {sequence}/{no_sequences}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Umpire Signal Data Collection', frame)
                cv2.waitKey(1000)
            else: 
                cv2.putText(frame, f'{action} | Sequence {sequence}/{no_sequences} | Frame {frame_num}/{sequence_length}', (15,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Umpire Signal Data Collection', frame)
            
            keypoints = extract_keypoints(detection_result)
            window.append(keypoints)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()
                
        npy_path = os.path.join(DATA_PATH, action, str(sequence))
        np.save(npy_path, np.array(window))
        
cap.release()
cv2.destroyAllWindows()
