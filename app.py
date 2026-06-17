import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import cv2
except ImportError:
    import os
    os.system("pip uninstall -y opencv-python opencv-contrib-python")
    os.system("pip install opencv-python-headless opencv-contrib-python-headless")
    import cv2
import numpy as np
import urllib.request
import os
import tempfile
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Umpire Signal Detection", layout="wide")
st.title("🏏 Real-Time Umpire Signal Detection")
st.markdown("Upload a video clip to automatically detect the umpire's signal.")

model_path = 'pose_landmarker_lite.task'
if not os.path.exists(model_path):
    with st.spinner("Downloading Pose Landmarker model..."):
        urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task', model_path)

@st.cache_resource
def load_detector():
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
        return vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        st.error(f"Failed to load MediaPipe PoseLandmarker: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

@st.cache_resource
def load_keras_model():
    try:
        return load_model('umpire_model.keras')
    except:
        return None

detector = load_detector()
model = load_keras_model()

if model is None:
    st.error("Model not found! Please train the model and save it as 'umpire_model.keras' in the root directory.")
    st.stop()

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

uploaded_file = st.file_uploader("Upload a Video", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    sequence = []
    current_action = "Idle"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)
        
        if detection_result.pose_landmarks:
            draw_landmarks(image, detection_result.pose_landmarks[0])
            
        keypoints = extract_keypoints(detection_result)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            action_idx = np.argmax(res)
            if res[action_idx] > threshold:
                current_action = ACTIONS[action_idx]
                
        cv2.rectangle(image, (0,0), (640, 50), (40, 40, 40), -1)
        cv2.putText(image, f'Signal: {current_action}', (10,35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        stframe.image(image, channels="RGB")
        
    cap.release()
    st.success("Video processing complete!")
