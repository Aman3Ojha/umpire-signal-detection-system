import cv2
import numpy as np
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model

model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
detector = vision.PoseLandmarker.create_from_options(options)

POSE_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)]

def draw_landmarks(image, landmarks):
    h, w, c = image.shape
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm, end_lm = landmarks[start_idx], landmarks[end_idx]
            cv2.line(image, (int(start_lm.x * w), int(start_lm.y * h)), (int(end_lm.x * w), int(end_lm.y * h)), (245, 66, 230), 2)
    for lm in landmarks:
        cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, (245, 117, 66), cv2.FILLED)

def extract_keypoints(detection_result):
    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]
        res = np.array([[lm.x, lm.y, lm.z, getattr(lm, 'visibility', 1.0)] for lm in landmarks]).flatten()
    else:
        res = np.zeros(33*4)
    return res

ACTIONS = np.array(['Out', 'Six', 'Four', 'Wide', 'No_Ball', 'Idle'])
sequence_length = 30
threshold = 0.8

print("Loading Umpire Detection Model...")
try:
    model = load_model('umpire_model.keras')
except Exception as e:
    print(f"Could not load 'umpire_model.keras'. Error: {e}")
    exit()

sequence = []
current_action = "Idle"

cap = cv2.VideoCapture(0)
print("Starting Webcam. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detection_result = detector.detect(mp_image)
    
    if detection_result.pose_landmarks:
        draw_landmarks(frame, detection_result.pose_landmarks[0])
    
    keypoints = extract_keypoints(detection_result)
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]
    
    if len(sequence) == sequence_length:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        action_idx = np.argmax(res)
        if res[action_idx] > threshold:
            current_action = ACTIONS[action_idx]
            
    cv2.rectangle(frame, (0,0), (640, 50), (40, 40, 40), -1)
    cv2.putText(frame, f'Signal: {current_action}', (10,35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Real-Time Umpire Signal Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
