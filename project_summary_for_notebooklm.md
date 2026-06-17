# Technical Report: Real-Time Umpire Signal Detection System
**Prepared for Presentation and Slide Generation (NotebookLM Compatible)**

---

## 1. Executive Summary & Project Goal
* **Objective:** Design, train, and deploy an automated, real-time computer vision and deep learning system that detects cricket umpire signals (Out, Six, Four, Wide, No Ball, Idle) from standard video clips.
* **Context:** Inspired by academic research in sports analytics, the system replaces manual tagging and tracking of signals with automated video comprehension.
* **Core Technologies:** Python, Streamlit, MediaPipe (Pose Landmarker API), TensorFlow/Keras (LSTM Networks), OpenCV, and GitHub.
* **Outcome:** A cloud-native web application deployed on Streamlit Community Cloud that handles live video uploads, processes frames to extract keypoints, draws real-time overlays, and predicts the umpire's signal with **94.4% accuracy**.

---

## 2. Technical System Architecture
The system operates as a sequential pipeline:
```
[ Video Input (.mp4/.mov) ] 
       │
       ▼
[ OpenCV Video Capture ] ──(Frame extraction)──► [ MediaPipe Pose Landmarker ]
                                                           │
                                                           ▼
                                                [ Extract 33 Landmarks ]
                                                [ 132 features / frame ]
                                                           │
                                                           ▼
                                                [ Sequence Buffer (30 Frames) ]
                                                           │
                                                           ▼
                                                [ Trained LSTM Model ]
                                                           │
                                                           ▼
                                                [ Predicted Class Probabilities ]
                                                           │
                                                           ▼
                                           [ Display overlay on Streamlit Frame ]
```

---

## 3. Data Collection and Representation
* **Landmark Extraction:** The system utilizes MediaPipe's Pose Landmarker. It tracks **33 distinct body joint landmarks** (shoulders, elbows, wrists, hips, etc.).
* **Feature Vector:** Each landmark has 4 dimensions: $(X, Y, Z)$ coordinates and a visibility/confidence score. This results in:
  $$\text{Feature Length} = 33 \times 4 = 132 \text{ numerical features per frame}$$
* **Temporal Windowing:** Actions occur over time rather than in static frames. The pipeline aggregates sequence windows of **30 frames** (approximately 1 second of video at 30fps) to detect motion dynamics.
* **Dataset Scope:** Collected multiple sample video sequences for 6 distinct categories:
  1. `Out` (index finger raised)
  2. `Six` (both arms raised vertically)
  3. `Four` (arm swept horizontally across the chest)
  4. `Wide` (both arms extended horizontally)
  5. `No_Ball` (one arm extended horizontally)
  6. `Idle` (standing posture, no signal)

---

## 4. LSTM Neural Network Design
To capture temporal relationships between frames, we implemented a Long Short-Term Memory (LSTM) recurrent neural network:
* **Input Layer:** Shape `(30, 132)` representing 30 sequential frames of 132 keypoints.
* **Layer 1: LSTM (64 Units, Return Sequences = True)** – Captures early spatial-temporal transitions.
* **Regularization: Dropout (20%)** – Prevents overfitting to the training samples.
* **Layer 2: LSTM (128 Units, Return Sequences = True)** – Builds a richer mid-level representation of the movement sequence.
* **Regularization: Dropout (20%)**
* **Layer 3: LSTM (64 Units, Return Sequences = False)** – Collapses the temporal dimension into a unified feature vector.
* **Regularization: Dropout (20%)**
* **Dense Layer (64 Units, ReLU Activation)** – Performs non-linear feature mapping.
* **Dense Output Layer (6 Units, Softmax Activation)** – Outputs the probability distribution across the 6 signal classes.
* **Optimization Parameters:** 
  * Optimizer: **Adam** (adaptive learning rate)
  * Loss Function: **Categorical Crossentropy**
  * Epochs: **200** (using model checkpointing to save the configuration with the highest validation accuracy)

---

## 5. Streamlit Cloud-Native Deployment Challenges & Solutions
Deploying OpenCV, MediaPipe, and TensorFlow to the Debian 13 (Trixie) Linux image on Streamlit Community Cloud required solving three major engineering roadblocks:

### A. Headless OpenCV & GUI Dependencies
* **Problem:** Standard `opencv-python` looks for desktop GUI display libraries (`libGL.so.1` etc.) which are absent in cloud instances, raising `ImportError` on startup.
* **Solution:** We configured the system to use `opencv-python-headless` and `opencv-contrib-python-headless` instead, and wrapped OpenCV loaders to dynamically swap libraries in memory if a failure is detected.

### B. MediaPipe C++ Bindings & Shared Libraries
* **Problem:** MediaPipe utilizes compiled C++ libraries loaded dynamically via `ctypes.CDLL`. This fails on bare Linux images due to missing system packages, generating unredacted `OSError` crashes.
* **Solution:** Developed a custom `packages.txt` to inject the necessary underlying libraries into the Debian package manager (`apt-get`) before Python starts:
  * `libgl1` - OpenGL support.
  * `libglib2.0-0t64` - GLib core utilities.
  * `libgles2` - GLES v2 support.
  * `libegl1` - Native platform graphics interface support.

### C. Cross-Environment Keras Deserialization Fallback
* **Problem:** Keras versions sometimes differ between local training environments (Windows) and cloud hosting environments (Debian). This causes Keras models saved as `.keras` format to crash during `load_model()` due to unrecognized metadata fields (such as `quantization_config`).
* **Solution:** Added a self-healing loader in the app logic. If a direct model load fails, the application automatically:
  1. Reconstructs the exact Sequential layer-by-layer architecture programmatically.
  2. Compiles the blank model structure.
  3. Loads the raw weights directly from the `.keras` model file, bypassing environment-specific configuration conflicts.

---

## 6. Key Features of the Streamlit Application
1. **Simple File Upload:** Supports drag-and-drop or browsing for `.mp4`, `.mov`, and `.avi` video clips.
2. **Real-Time Skeleton Overlay:** Draws pose landmark joints and links directly on the video frames in real time using OpenCV.
3. **Rolling Prediction Buffer:** Evaluates predictions over the latest 30 frames and prints the detected umpire signal dynamically at the top of the video container.
4. **Resilient System Design:** Gracefully handles missing dependencies and model deserialization conflicts automatically.
