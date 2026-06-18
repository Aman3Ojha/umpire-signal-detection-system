# 🏏 Real-Time Umpire Signal Detection

An AI-powered, real-time cricket umpire signal detection system. Inspired by the academic paper *"Real-Time Umpire Signal Detection in Cricket: A Hybrid Deep Learning Solution"*, this project utilizes a hybrid approach combining spatial feature extraction (pose estimation) and temporal sequence modeling (LSTM) to classify actions dynamically.

## 📊 Project Presentation
* **[Umpire Signal Detection Presentation (PPTX)](./Umpire_Signal_Detection_Presentation.pptx)** — Widescreen technical slide deck detailing model architectures, features, and deployment configurations.

## 🌟 Features
- **Real-Time Detection**: Analyzes live webcam feed to predict umpire signals instantly.
- **Hybrid Architecture**: Uses Google's **MediaPipe Tasks API** for blazingly fast 33-point skeletal landmark extraction, coupled with a custom **TensorFlow/Keras LSTM** model for time-series action recognition.
- **CPU Friendly**: Designed to be lightweight and run efficiently on standard CPUs without requiring expensive GPU hardware.
- **Interactive Web App**: Includes a modern **Streamlit** dashboard to upload and process pre-recorded cricket video clips.

## 🛠️ Tech Stack
- **Python 3**
- **OpenCV** (Computer Vision & Video Processing)
- **MediaPipe** (Pose/Skeleton Landmark Extraction)
- **TensorFlow / Keras** (Deep Learning / LSTM)
- **Streamlit** (Web Interface)
- **NumPy & Scikit-Learn** (Data Handling)

## 🚀 Installation & Setup

1. **Clone or Download the Repository**
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Mac/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎥 Usage Guide

The project is split into 4 distinct phases:

### 1. Data Collection (`data_collection.py`)
Run this script to build your own dataset. It uses your webcam to record you performing the umpire signals and saves the skeletal data as `.npy` arrays.
```bash
python data_collection.py
```
*Follow the on-screen instructions. Press `s` to begin recording the sequence for each signal.*

### 2. Model Training (`train_model.py`)
Once your data is collected in the `data/` folder, train the LSTM network.
```bash
python train_model.py
```
*The script will process the data, train the neural network, and save the best weights to `umpire_model.keras`.*

### 3. Live Inference (`inference.py`)
Test the AI in real-time. Stand back and perform the signals in front of your camera.
```bash
python inference.py
```

### 4. Web Dashboard (`app.py`)
Launch the Streamlit web application to process uploaded video files instead of live webcam feeds.
```bash
streamlit run app.py
```
*Navigate to `http://localhost:8501` in your browser to interact with the dashboard.*

---

*Note: On your first run, the system will automatically download a small `pose_landmarker_lite.task` model file (approx. 3MB) required by MediaPipe.*

## 📚 Citation & References

This project takes reference from the following research paper:

```bibtex
@article{kaur2025realtime,
  title={Real-Time Umpire Signal Detection in Cricket: A Hybrid Deep Learning Solution},
  author={Kaur, Baljinder and Rani, Diksha and Jayaprakash, Lithiga and Akshita and kaur, Ramneet and kumar, Aman and kumari, Shivani and kumar, Ansh},
  journal={Journal of Information Systems Engineering and Management},
  volume={10},
  number={54s},
  year={2025},
  issn={2468-4376},
  url={https://www.jisem-journal.com/}
}
```

### Paper Metadata
* **Title:** Real-Time Umpire Signal Detection in Cricket: A Hybrid Deep Learning Solution
* **Journal:** *Journal of Information Systems Engineering and Management* (2025, 10(54s))
* **e-ISSN:** 2468-4376
* **Official Publisher URL:** [https://www.jisem-journal.com/](https://www.jisem-journal.com/)
* **Authors:**
  1. **Baljinder Kaur** (University Institute of Liberal Arts, Chandigarh University, Mohali, Punjab)
  2. **Diksha Rani** (University Institute of Liberal Arts, Chandigarh University, Mohali, Punjab)
  3. **Lithiga Jayaprakash** (AIT-CSE (AIML), Chandigarh University, Mohali, Punjab)
  4. **Akshita** (AIT-CSE (AIML), Chandigarh University, Mohali, Punjab)
  5. **Ramneet Kaur** (CSE, Chandigarh University, Mohali, Punjab)
  6. **Aman Kumar** (CSE, Chandigarh University, Mohali, Punjab)
  7. **Shivani Kumari** (CSE, Chandigarh University, Mohali, Punjab)
  8. **Ansh Kumar** (CSE, Chandigarh University, Mohali, Punjab)

